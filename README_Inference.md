**Suppose we want to reconstruct neurons from a 512×512×512 voxel 3D fruit fly brain volume.**


## 🧩 The 8 Core Components — Organizational Roles

1. **`request` (Blueprint / Contract)**  
   A Protobuf configuration file.  
   It specifies:
   - Where the data is (HDF5 path)
   - Where the model weights are
   - Thresholds and inference parameters

2. **`counters` (Performance Dashboard)**  
   A distributed system of timers and counters tracking:
   - GPU idle time
   - Inference time
   - Segmented voxel count
   - Growth statistics

3. **`Runner` (Chief Engineer)**  
   Reads the `request`, manages global resources, divides the 512³ region into work blocks, and assigns jobs.

4. **`executor` (Central GPU Factory)**  
   A background server thread.  
   It batches incoming inference requests and runs them on the GPU.

5. **`canvases` (Project Registry)**  
   A dictionary `{corner: canvas}` maintained by `Runner`.  
   Keeps track of active subvolume jobs to prevent premature garbage collection.

6. **`canvas` (3D Construction Crew)**  
   Manages one specific 512³ subvolume.  
   Holds its own segmentation buffer and communicates with the GPU.

7. **`seed_policy` (Exploration Scout)**  
   For example: `PolicyPeaks`.  
   Identifies safe starting seed points within the 3D volume.

8. **`movement_policy` (Tactical Navigator)**  
   For example: `FaceMaxMovementPolicy`.  
   Determines where the Field-of-View should move next during object growth.

---

# 🎬 Full-System Execution: Reconstructing a 512³ Brain Volume

Let’s follow the real data flow step by step.

---

## Phase 1 — System Initialization  
`request → Runner → executor`

1. **Command Issued**  
   You launch inference via CLI.  
   The configuration is parsed into a `request` object.

2. **Runner Instantiated**  
   `Runner.start(request)` is called.

3. **GPU Engine Ignition**
   - The PyTorch model is loaded onto the GPU.
   - A `ThreadingBatchExecutor` is created.
   - `executor.start_server()` starts the background inference thread.

   The executor now enters an infinite loop, waiting for tasks.

4. **Mount Large Volume**
   The HDF5 file containing the brain volume is opened lazily (not fully loaded into RAM).

---

## Phase 2 — Allocating the 512³ Subvolume  
`Runner → canvas → canvases`

1. **Subvolume Requested**
   ```python
   runner.run(corner=(0,0,0), subvol_size=(512,512,512))
   ```

2. **Data Extracted**
   The 512³ voxel block is read and normalized.

3. **Canvas Created**
   A `canvas` instance is created:

   * Holds the 512³ image
   * Owns a segmentation buffer
   * Has a communication client to the GPU executor
   * Has its own `counters`

4. **Registered in Registry**

   ```python
   self.canvases[(0,0,0)] = canvas
   ```

---

## Phase 3 — Seed Discovery

`canvas → seed_policy`

1. **Control Passed to Canvas**

   ```python
   canvas.segment_all(seed_policy=PolicyPeaks)
   ```

2. **Full-Volume Analysis**
   `seed_policy` runs classical CV:

   * Sobel edge detection
   * Euclidean distance transform (EDT)
   * Local maxima detection

   Suppose it finds 10,000 candidate seed points.

3. **Seeds Yielded Sequentially**
   Seeds are returned one-by-one via an iterator.

---

## Phase 4 — Neuron Growth and GPU Interaction

`canvas → movement_policy → executor`

Now the first seed begins growing.

1. **Seed Enqueued**
   The seed position is pushed into `movement_policy`.

2. **Local FOV Extraction**
   A 33×33×33 cube is cropped around the current location.

3. **GPU Inference Request**

   ```python
   logits = executor.predict(seed_logits, image_patch)
   ```

   The thread blocks until inference completes.

4. **Executor Batching**

   * Requests accumulate in a queue.
   * Batched into size 64 (or timeout triggers execution).
   * Forward pass runs on GPU.

5. **Results Returned**
   The logits are returned to the canvas.

6. **Growth Decision**
   `movement_policy` inspects the 6 faces of the predicted cube.
   New candidate positions are pushed into its priority queue.

7. **Repeat**
   The neuron grows like a 3D flood-filling snake until no valid expansion remains.

---

## Phase 5 — Object Finalization

`canvas → counters`

1. **No Further Growth**
   Movement queue becomes empty.

2. **Validation**

   * Check object size
   * Check confidence
   * Assign segmentation ID

3. **Metrics Updated**
   `counters` record:

   * Voxel count
   * Time spent
   * Inference latency

---

## Phase 6 — Subvolume Completion

`canvas → Runner → canvases`

1. **All Seeds Processed**
   10,000 seeds evaluated.

2. **Results Saved**
   Segmentation volume written to disk.

3. **Canvas Deregistered**

   ```python
   del self.canvases[(0,0,0)]
   ```

   Memory is freed.

---

# 🚀 The True Power: Concurrent Multi-Block Inference

If you launch 4 parallel threads:

```python
runner.run(corner=A)
runner.run(corner=B)
runner.run(corner=C)
runner.run(corner=D)
```

Then:

* 4 independent `canvas` objects exist simultaneously.
* 4 separate `seed_policy` and `movement_policy` instances operate independently.
* All send inference requests into the **same executor queue**.
* The executor efficiently batches them.
* GPU utilization approaches peak efficiency.

This architecture is designed for industrial-scale throughput:

* Many independent subvolumes
* Shared centralized GPU compute
* CPU threads saturating inference pipelines
