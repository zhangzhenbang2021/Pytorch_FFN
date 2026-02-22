# FFN-PyTorch Inference Module

This module implements the full FFN inference pipeline: model execution, segmentation state management, FOV movement policy, seed policy, and result I/O.

---

## Module Overview

### `executor.py` — Batch Inference Executor

Uses a **client–server threading architecture** for efficient GPU use:

- **`ExecutorInterface`**: Communication channel with input queue and output queue dict
- **`ThreadingExecutorClient`**: Client that submits inference requests and waits for results
- **`ThreadingBatchExecutor`**: Server holding the PyTorch model and GPU resources
  - Collects requests from multiple clients into batches
  - Runs batch inference (`@torch.no_grad()`)
  - Handles BZYX1↔BCZYX tensor format conversion
  - Dispatches results back to each client’s output queue

Protocol:
- Positive integer N: client N registers
- Negative integer -N-1: client N unregisters
- String `"exit"`: request server shutdown
- Tuple `(client_id, seed, image, fetches)`: inference request

### `inference.py` — Canvas (Inference State)

`Canvas` is the core class for inference, maintaining full segmentation state for a subvolume:

**Data structures:**
- `seed`: Current working mask (logit space), shape `[Z, Y, X]`, NaN = uninitialized
- `segmentation`: Segmentation result, integer IDs (0=background, >0=object, <0=excluded)
- `seg_prob`: Optional object probability map (quantized to uint8)
- `origins`: Per-object metadata (start position, iterations, wall time)

**Flow:**
```
segment_all()
  │
  ├─→ seed_policy generates candidate seed positions
  │     │
  │     ▼
  ├─→ is_valid_pos() checks position validity
  │     │
  │     ▼
  ├─→ segment_at(pos) segments from that position
  │     │
  │     ├─→ init_seed(pos) initialize seed
  │     │
  │     ├─→ movement_policy iterates FOV positions
  │     │     │
  │     │     ▼
  │     ├─→ update_at(pos) single-step prediction
  │     │     │
  │     │     ├─→ predict() → executor → network inference
  │     │     │
  │     │     └─→ update seed[sel] = logits
  │     │
  │     └─→ movement_policy.update(pred, pos)
  │
  ├─→ Threshold and min-size checks
  │
  └─→ Assign segment ID → segmentation[mask] = sid
```

**Checkpointing:** Canvas supports periodic save/restore of inference state (NPZ), so long runs can be interrupted and resumed.

### `runner.py` — Runner (Orchestration)

`Runner` orchestrates the full inference run:

1. **Init**: Load PyTorch model, open data volume, start executor
2. **Subvolumes**: For each subvolume, create Canvas, run segmentation, save results
3. **I/O**: Alignment, cropping, checkpoints, segmentation save

Key methods:
- `start(request)`: Initialize from InferenceRequest protobuf
- `make_canvas(corner, size)`: Build Canvas (data load and alignment)
- `run(corner, size)`: Run full subvolume inference
- `save_segmentation()`: Save segmentation and probability map

### `movement.py` — FOV Movement Policy

- **`FaceMaxMovementPolicy`**: Standard movement policy
  - Maintains a priority queue of scored positions
  - Picks the highest-scoring neighbor
  - Uses `get_scored_move_offsets()` to score all possible moves
  - Already-segmented regions are marked non-movable

- **`MovementRestrictor`**: Restricts movement in certain regions (mask-based)

### `seed.py` — Seed Policies

Decide where to start new segments:

| Policy | Description |
|--------|-------------|
| `PolicyPeaks` | Local peaks of distance transform (object centers) |
| `PolicyPeaks2d` | 2D version of peak policy |
| `PolicyGrid3d` | Uniform sampling on a 3D grid |
| `PolicyInvertOrigins` | Reverse search from existing object origins |

### `storage.py` — I/O and Storage

- **Volume loading**: HDF5 and TensorStore
- **Atomic writes**: Avoid half-written files on interrupt
- **Segmentation save/load**: NPZ (segmentation, origins, counters)
- **Probability map**: Quantized uint8 (255 levels)
- **Mask building**: Coordinate expressions, volume channels, inversion, etc.

### `inference_utils.py` — Utilities

- **`Counters`**: Thread-safe counter container (replaces TF variables)
- **`StatCounter`**: Single counter with Increment/Set/IncrementBy
- **`timer_counter`**: Context manager for call count and elapsed time
- **`TimedIter`**: Wraps an iterator with timing
- **`match_histogram`**: Histogram matching for preprocessing

### `segmentation.py` — Post-processing

- **`clear_dust`**: Remove objects below a size threshold
- **`reduce_id_bits`**: Choose smallest integer type for segment IDs
- **`clean_up`**: Connected components + dust removal
- **`split_segmentation_by_intersection`**: Intersection of two segmentations

### `align.py` — Alignment

- **`Alignment`**: Base class (identity), coordinate transform between subvolumes
- **`Aligner`**: Base class for alignment generators

---

## Inference Data Flow

```
                    ┌────────────────────────┐
                    │   InferenceRequest     │  (protobuf config)
                    │  - model_checkpoint    │
                    │  - image volume        │
                    │  - bounding box        │
                    │  - inference options   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      Runner.start()    │
                    │  - Load PyTorch model  │
                    │  - Open data volume    │
                    │  - Start Executor      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Runner.run()        │
                    │  - Load image subvol   │
                    │  - Create Canvas       │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
    ┌──────────────┐   ┌──────────────┐    ┌──────────────┐
    │  SeedPolicy  │   │    Canvas    │    │   Executor   │
    │  seed pos    │   │  seg state   │    │  GPU batch   │
    └──────┬───────┘   └──────┬───────┘    └──────┬───────┘
           │                  │                    │
           │    segment_all() │     predict()      │
           └──────────────────┼────────────────────┘
                              │
                              ▼
                    ┌────────────────────────┐
                    │  save_segmentation()   │
                    │  - Segmentation (NPZ)  │
                    │  - Probability (NPZ)    │
                    │  - Counter stats        │
                    └────────────────────────┘
```

---

## Output File Layout

Under `segmentation_output_dir`:

```
output_dir/
├── {z}/{y}/
│   ├── seg-{x}_{y}_{z}.npz     # Segmentation
│   │   ├── segmentation          # int32 labels
│   │   ├── origins               # Per-object metadata
│   │   ├── request               # Serialized inference request
│   │   ├── counters              # JSON counters
│   │   └── overlaps              # Overlap info
│   │
│   ├── seg-{x}_{y}_{z}.prob     # Probability map
│   │   └── qprob                 # uint8 quantized prob
│   │
│   └── seg-{x}_{y}_{z}.cpoint   # Checkpoint (removed when done)
│
└── counters.txt                  # Global counters
```
