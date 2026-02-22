# FFN-PyTorch Training Module

This module implements the full FFN training pipeline: model definition, data loading, training sample generation, augmentation, and metric tracking.

---

## Module Overview

### `model.py` — Model Base

Defines two core abstractions:

- **`ModelInfo`**: Dataclass for network geometry
  - `deltas`: FOV step size (x, y, z)
  - `pred_mask_size`: Prediction mask size
  - `input_seed_size`: Input seed size
  - `input_image_size`: Input image size

- **`FFNModel(nn.Module)`**: Base class for all FFN models
  - `forward(image, seed) → logits`: Subclasses must implement
  - `update_seed(seed, update)`: Add network output to seed
  - `compute_loss(logits, labels, weights)`: Weighted BCE loss
  - `shifts`: List of all FOV move directions (26-neighborhood)

### `models/convstack_3d.py` — 3D Conv Stack

Implements the standard architecture from the paper:

```
Input: concat(image, seed)  →  [B, 2, Z, Y, X]
     ↓
  Conv3d(2→f, 3×3×3) + ReLU
  Conv3d(f→f, 3×3×3)           ← input_block
     ↓
  ┌─→ ReLU → Conv3d(3×3×3) → ReLU → Conv3d(3×3×3) ─┐
  │                                                    │
  └────────────── + (residual) ←───────────────────────┘
     ↓  (repeat depth-1 times)
  ReLU → Conv3d(f→1, 1×1×1)   ← output_conv (logit update)
     ↓
  update_seed(seed, logit_update)  → [B, 1, Z, Y, X]
```

- **depth**: Number of residual blocks (default 9)
- **features**: Channels per layer (default 32)
- **Weight init**: Truncated normal (std=0.01)

### `inputs.py` — Data Loading

- **`CoordinateLoader`**: Load training coordinates from HDF5 or TFRecord
- **`load_from_numpylike`**: Extract subvolumes from HDF5 datasets
- **`load_example`**: Build full training sample (image + labels + weights)
- **`soften_labels`**: Boolean labels → soft labels (0.05 / 0.95)
- **`offset_and_scale_patches`**: Image normalization

Data flow:
```
Coordinate file (HDF5) → CoordinateLoader → load_from_numpylike → augment → normalize
```

### `examples.py` — Training Sample Generation

Simulates FOV movement during inference to produce training samples:

- **`get_example`**: Core generator; at each training position, simulates multiple FOV steps
- **`fixed_offsets`**: Uses a fixed list of moves (from `model.shifts`)
- **`max_pred_offsets`**: Simulates inference-time max-prediction move policy
- **`no_offsets`**: No move, predict only at center
- **`BatchExampleIter`**: Multi-threaded batch generator
  - `update_seeds()`: Feed network predictions back into seeds (simulates iterative update)

### `augmentation.py` — Data Augmentation

Augmentations for serial-section EM (ssEM):

| Method | Description | Dimensions |
|--------|-------------|------------|
| `PermuteAndReflect` | Axis permutation and flip | Global |
| `elastic_warp` | Elastic deformation | Random Z slices |
| `affine_transform` | Affine (rotate/scale/shear) | Random Z slices |
| `misalignment` | Slice alignment error | Z shift/translate |
| `missing_section` | Missing slices | Random Z slices |
| `out_of_focus_section` | Out-of-focus slices | Random Z (Gaussian blur) |
| `grayscale_perturb` | Brightness/contrast/gamma | Global or per-slice |

### `mask.py` — Mask Utilities

- **`crop`**: Crop subvolume at center offset
- **`crop_and_pad`**: Crop with optional zero-pad to target size
- **`update_at`**: Write values at given offset
- **`make_seed`**: Create center single-voxel seed array

### `tracker.py` — Metric Tracking

- **`EvalTracker`**: Tracks evaluation during training
  - Move stats: correct/missed/false move ratios
  - Segmentation quality: TP/TN/FP/FN → precision/recall/F1
  - Loss: per-patch BCE
  - Visualization: label/prediction/weights comparison

### `optimizer.py` — Optimizer Config

Supported: SGD, Momentum SGD, Adam, Adagrad, RMSProp

### `import_util.py` — Dynamic Import

`import_symbol("convstack_3d.ConvStack3DFFNModel")` loads the model class dynamically, defaulting to the `ffn_pytorch.training.models` package.

---

## Training Data Flow

```
                        ┌──────────────────┐
                        │  HDF5 coords     │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ CoordinateLoader │  (random sample coords)
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Image   │ │  Label   │ │ Augment  │
              │ volume   │ │ volume   │ │          │
              │ (HDF5)   │ │ (HDF5)   │ │          │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   get_example    │  (simulate FOV moves)
                        │   generator      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ BatchExampleIter │  (multi-thread batching)
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              seed [B,1,Z,Y,X]  image  labels+weights
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  PyTorch model   │  (forward + backward + opt)
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  update_seeds    │  (predictions → seeds)
                        └──────────────────┘
```
