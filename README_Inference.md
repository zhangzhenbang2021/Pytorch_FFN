
**Suppose we are about to reconstruct neurons from a 512×512×512 voxel 3D image block of a fruit fly brain.**

Let’s anchor the **“job roles”** of these 8 core components in one sentence each:

1. **`request` (Blueprint / Contract)**: The Protobuf configuration file. It contains where the blueprint is (HDF5 path), where the AI model is, and what the passing thresholds are.
2. **`counters` (Performance Monitoring Dashboard)**: Stopwatch and step counters everywhere, recording how long the GPU idled and how much volume successfully grew.
3. **`Runner` (Chief Project Commander)**: Reads the `request`, controls global resources, partitions the 512×512×512 region, and assigns construction teams.
4. **`executor` (Central GPU Compute Workshop)**: A background daemon thread. The sole worker responsible for batching scattered tasks, sending them to the GPU, and distributing the results back.
5. **`canvases` (Project Registry Book)**: A dictionary `{corner: canvas}` held by `Runner`, recording which blocks are currently under construction to prevent them from being garbage-collected.
6. **`canvas` (Frontline 3D Construction Team)**: Responsible for the specific 512×512×512 block. It owns its own 3D matrix and handles neuron coloring.
7. **`seed_policy` (Air-Dropped Scout)**: For example, `PolicyPeaks`. Responsible for scanning the entire 512×512×512 map to find the thickest, safest “center points” as growth starting positions.
8. **`movement_policy` (Tactical Navigation Unit)**: For example, `FaceMax`. After a starting point is determined, it guides the AI on which direction (up, down, left, right, etc.) to explore next within the 33×33×33 local field of view.

---

## 🎬 Full-System Simulation: The Journey of Reconstructing a 512×512×512 Brain Block

Let’s strictly follow the code’s data flow and retrace this astonishing journey.

### Phase 1: Command Center Initialization and Central Workshop Ignition (`request` → `Runner` → `executor`)

1. **Issuing the Command**: You pass in a configuration file via the command line. It is parsed into a `request` object. It specifies the HDF5 image path to process, the model weights path, and `batch_size=64`.
2. **Chief Commander Takes Position**: The `Runner` instance is created. It calls the `start(request)` method.
3. **Ignition**: `Runner` examines the `request`, loads the PyTorch model onto the GPU. Then it instantiates the `executor` (`ThreadingBatchExecutor`) with this model and calls `executor.start_server()`.  
   *At this moment, the background `executor` thread begins spinning in a `while` loop, staring at an empty queue. The GPU is idling in standby mode.*

4. **Mounting the Master Map**: `Runner` lazily opens the massive HDF5 file containing the entire fruit fly brain using `decorated_volume`, but does not load it into memory.

---

### Phase 2: Contracting the 512×512×512 Block (`Runner` → `canvas` → `canvases`)

1. **Claiming Territory**: The main program calls `runner.run(corner=(0,0,0), subvol_size=(512,512,512))`.
2. **Material Extraction**: `Runner` calls `make_canvas`, truly reading the 512×512×512 grayscale matrix from HDF5 and normalizing it.
3. **Allocating the Canvas**: A `canvas` object is created. It holds this 512×512×512 image, a walkie-talkie (`_exec_client` used to call the GPU), and its own performance dashboard `counters`.
4. **Project Registration**: `Runner` records it in the registry: `self.canvases[(0,0,0)] = canvas`.

---

### Phase 3: Air-Dropped Exploration and Seed Discovery (`canvas` → `seed_policy`)

1. **Transfer of Command**: `Runner` executes `canvas.segment_all(seed_policy=PolicyPeaks)`. At this point, it steps back and hands the stage entirely to `canvas`.
2. **Full-Map Scanning**: `canvas` activates the `seed_policy`.  
   - `seed_policy` receives the 512×512×512 image and executes computationally intensive classical CV algorithms (Sobel edge detection + Euclidean Distance Transform).
   - It searches the mountainous distance map for local maxima (peaks).
   - Suppose it finds 10,000 safe “seed coordinates” (for example, one is `pos = (100, 200, 300)`).

3. **Queueing for Dispatch**: In the loop `for pos in TimedIter(self.seed_policy...)`, `seed_policy` begins delivering these 10,000 seeds one by one to `canvas` via `__next__`.

---

### Phase 4: Explosive Cell Growth and GPU Fire Support (`canvas` → `movement_policy` → `executor`)

This is the most intense and complex phase of data flow. Now `canvas` receives the first seed `(100, 200, 300)` and calls `segment_at` to reconstruct this neuron.

1. **Activate Navigator**: `canvas` pushes the starting point `(100, 200, 300)` into the priority queue of `movement_policy`. The big loop `for pos in self.movement_policy` begins turning!
2. **Crop and Call**:
   - `canvas` moves to `(100, 200, 300)` and calls `update_at`.
   - It crops a 33×33×33 tiny field of view `img` from the 512×512×512 master map.
   - `canvas` picks up the walkie-talkie: `self._exec_client.predict(logit_seed, img)`.  
     **The current thread instantly goes to sleep!**

3. **Workshop Batching and GPU Showdown**:
   - The cropped 33×33×33 block flies into the background `executor` queue.
   - `executor` sees only 1 task, waits less than 5 ms (anti-starvation trigger), and converts it into a PyTorch tensor: `[1, 1, 33, 33, 33]`.
   - **GPU fires at full capacity**: `model(image_t, seed_t)`.
   - After computation, `executor` sends the predicted 33×33×33 logits back to the sleeping caller.

4. **Decision for the Next Move**:
   - The awakened `canvas` paints the predicted region onto the 512×512×512 canvas.
   - It then hands the logits to `movement_policy`.
   - `movement_policy` uses the `FaceMax` logic to inspect the 6 faces of the 33×33×33 cube.  
     It finds that “forward” and “upward” are valid extensions of the neuron, computes two new coordinates `(100, 208, 300)` and `(108, 200, 300)`, and pushes them into its queue.

5. **Repeat**:  
   `for pos in self.movement_policy` continues popping new coordinates.  
   Crop → call `executor` → GPU inference → `movement_policy` finds new faces.  
   **This neuron grows like a 3D snake within the 512×512×512 space.**

---

### Phase 5: Completion and Quality Inspection (`canvas` → `counters`)

1. **Dead End Reached**: When the neuron hits the boundary or all faces fall below `move_threshold`, the loop ends.
2. **Inspection and Registration**:
   - `canvas` checks: Is the object large enough? Is the seed confident?
   - If valid, assign a unique ID (e.g., `sid=1`) and paint all its voxels with ID 1.
   - `counters` logs performance:  
     - `voxels-segmented` increases by 50,000  
     - `valid-time-ms` increases by 1200 ms

---

### Phase 6: Full Region Completion and Cleanup (`canvas` → `Runner` → `canvases`)

1. **All Seeds Consumed**:  
   `canvas` continues requesting seeds `(150, 250, 350)` and repeats phases 4–5, growing `sid=2`, `sid=3`, etc., until all 10,000 seeds are processed.
2. **Canvas Returned**:  
   `canvas.segment_all` completes and returns control to `Runner`.
3. **Compression and Save**:  
   `Runner` takes the fully colored 512×512×512 segmentation matrix, realigns coordinates, and saves it as `.npz`.
4. **Memory Release**:  
   `Runner` executes `del self.canvases[(0,0,0)]`.  
   The `canvas` and its hundreds of MB of memory are garbage-collected instantly.

---

## 💡 The Ultimate Form of Concurrency

The above describes single-threaded execution. But the true power of FFN lies in concurrency:

If your main program launches **4 CPU threads simultaneously calling `runner.run()`**, each with different 512×512×512 coordinates:

- `canvases` will hold 4 independent construction crews.
- Each has its own `seed_policy` and `movement_policy`.
- They fire 33×33×33 crops into the same `executor` queue like machine guns.
- The `executor` quickly fills a batch of 64 and feeds the GPU.
- GPU utilization reaches peak efficiency, and `counters` show optimal `inference_ms`.




