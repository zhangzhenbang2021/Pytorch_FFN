### I. Relationship of the Three Components: From "Static Cropping" to "Batch Packing"

#### 1. `_load_example` (Raw Material Supplier)

* **What it is**: A preset "box" with all parameters (crop size, augmentation, brightness normalization).
* **What it does**: Each call loads from the HDF5 file **one static 3D block** (49×49×25) centered on a target.
* **Characteristic**: It is **static**. The field of view (FOV) does not move inside this function.

#### 2. `_make_ffn_example` (Single Workstation)

* **What it is**: It assembles `_load_example` (raw material), `policy_fn` (move instructions), and init parameters into a **stateful generator** (the function with `while True` and `yield`).
* **What it does**: It takes one 3D block, plants a seed of 0.95 at the center, then **moves the FOV step by step** according to `policy_fn`, yielding a small local patch (33×33×17) at each step.
* **Characteristic**: It is **dynamic**, representing one neuron’s full "exploration lifecycle."

#### 3. `batch_it` (Batch Supervisor)

* **What it is**: `BatchExampleIter` wraps the single workstation `_make_ffn_example` and starts **FLAGS.batch_size** (e.g. 4) **background threads**.
* **What it does**: It watches those workstations. When you ask for data (`next(batch_it)`), it **concatenates** the current patches from all workstations into a standard batch tensor for the GPU.

---

### II. How Does the Updated Seed Flow? (Core Loop)

This answers: **How is the data produced by the network written back?**

Recall the `__next__` inside `BatchExampleIter`:

```python
seeds, patches, labels, weights = next(self._batch_generator)
self._seeds = seeds  # 👈 The key is here!
```

When `batch_it` yields data to the main program, it stores **references** to those local FOV regions in `self._seeds`.

**Update loop:**

1. GPU finishes forward pass and produces `logits`.
2. Main program calls `batch_it.update_seeds(updated_seed_np)`.
3. `batch_it` uses the saved `self._seeds` references to **overwrite/accumulate** the GPU `logits` onto the four background threads’ large canvases.
4. On the next `next(batch_it)`, the patches from the background threads already carry the network’s latest "ink."

---

### III. Does the Updated Seed Affect the Movement Policy (`policy_fn`)?

**Yes—absolutely and critically.** This is why FFN (flood-filling) is called "flood."

Here `policy_fn` is `fixed_offsets`. Updated seed affects it like this:

Inside the function used by `fixed_offsets`:

```python
valid_move = seed[:, center_offset_position] >= seed_threshold
```

**What happens?**

1. Suppose the policy tests a right move: `off = (8, 0, 0)`.
2. It **directly checks the current canvas (seed)**: at that right-shifted position, did the network’s last prediction reach the threshold (e.g. 0.9)?
3. **If yes (`valid_move = True`)**: The policy `yield off`, telling the program to "step right and crop a new patch for training."
4. **If no (`valid_move = False`)**: The policy skips this direction and checks the next.

**Conclusion:**  
The updated seed is like a **flashlight beam** the model casts in the dark. The movement policy (`policy_fn`) is like someone who only steps where the beam shines. If the network does not paint high probability on the right when updating the seed, the policy **will not** move the FOV to the right. The flow of training data is entirely driven by the network’s own output.

---

We will unfold the system in the order **"architecture → data birth → training forward → closed-loop update → policy pathfinding."**

---

### Part 1: The Three Core Components and the Data-Loading Pipeline

In the code, three functions/dicts are nested like "Russian dolls" and together form the FFN data assembly line:

#### 1. `policy_map`: The Movement Policy "Armory"

```python
policy_map = {
    'fixed': partial(examples.fixed_offsets, fov_shifts=fov_shifts, threshold=special.logit(FLAGS.threshold)),
    'max_pred_moves': partial(...)
}
policy_fn = policy_map[FLAGS.fov_policy]
```

* **Logic**: Python’s `functools.partial` **bakes in** config (e.g. `fov_shifts`, threshold as logit of 0.9) into `fixed_offsets`.
* **Result**: `policy_fn` becomes a "plug-and-play" navigator that only needs `(info, seed, full_labels, eval_tracker)` to run.

#### 2. `_load_example`: The "Miner" (Static Data Loading)

```python
def _load_example():
    return inputs.load_example(coord_loader, label_volume_map, ...)
```

* **Logic**: A closure that encapsulates all context needed to read from HDF5.
* **What it does**: Each call takes a point from the global `coord_loader`, **extracts one subvolume** from the raw volume, and uses `soften_labels(lom)` to keep only the single neuron passing through the center, softening labels to foreground/background (e.g. 0.05 / 0.95).

#### 3. `_make_ffn_example`: The "Single Workstation" (Dynamic Generation)

```python
def _make_ffn_example():
    return examples.get_example(_load_example, eval_tracker_obj, info, policy_fn, ...)
```

* **Logic**: It combines the "miner" (`_load_example`) and the "navigator" (`policy_fn`) and passes them to `get_example`.
* **Result**: `_make_ffn_example` becomes a **generator factory** that keeps yielding **local FOV tensors**. `BatchExampleIter` then runs FLAGS.batch_size (e.g. 4) workstations in parallel and batches their outputs.

---

### Part 2: Lifecycle of One Block and the Inner Move Loop

Inside `get_example`, here is how one large block is used until it is "exhausted."

#### Phase A: Birth of the Large Canvas

```python
while True:
    ex = load_example()  # Load full 49×49×25 image and labels
    full_patches, full_labels, ... = ex

    # Create [1, 25, 49, 49, 1] seed canvas
    seed = special.logit(mask.make_seed(seed_shape, 1, pad=seed_pad))
```

* The outer `while True` is the cycle over neurons.
* `mask.make_seed` fills the space with 0.05 and places 0.95 at the center.
* `special.logit` converts probabilities to log-odds (background ≈ -2.94, center ≈ +2.94). The **seed is planted**.

#### Phase B: Inner Loop Movement Policy

The code then enters the critical exploration phase:

```python
for off in get_offsets(info, seed, full_labels, eval_tracker):
```

Here `get_offsets` is the bound `fixed_offsets`. Its strategy is rigid but efficient:

1. **Init**: `fixed_offsets` has a fixed list `fov_shifts` (e.g. [(-8,0,0), (8,0,0), ...]) plus the mandatory origin (0,0,0).
2. **`_eval_move` (single-step lookahead)**: For each offset `off_xyz`, it does one thing: **on the large canvas `seed`, check whether the predicted probability at the move-target center ≥ threshold.**
```python
valid_move = seed[:, seed.shape[1]//2 + off_xyz[2], ...] >= seed_threshold
```
3. **Success → yield**: If `valid_move` is True, the policy yields that offset.
4. **Local crop**: Back in `get_example`, the program uses this `off` and `mask.crop_and_pad` to crop the current FOV (e.g. 33×33×17) from the large block and yields `predicted`, `patches`, `labels` to the training loop.
* *Note: The inner loop is **frozen** here until the outer code writes the new prediction back to `seed` (see Part 4).*

#### Phase C: Exhaustion

As the main loop keeps requesting data, the generator wakes, checks the next direction, wakes again, etc., until the `fov_shifts` list in `fixed_offsets` is **fully traversed**. Then the `for off in get_offsets(...)` loop ends; that neuron’s local structure in the block is done. The program calls `eval_tracker.add_patch(...)` to record quality, then the outer `while True` continues, discards the old block, and calls `load_example()` to fetch a new coordinate.

---

### Part 3: Main Program Receives Data, Forward Pass, and Loss

Data leaves the background generator and enters the main training loop (e.g. `train_ffn.py`).

#### 1. Dimension Conversion (Model Input)

```python
seed_np, patches_np, labels_np, weights_np = next(batch_it)
```

You get data from 4 threads concatenated, shape **`[4, 25, 49, 49, 1]` (NDHWC)**. PyTorch 3D conv expects **`[Batch, Channel, Depth, Height, Width]` (NCDHW)**. So a transpose is needed:

```python
seed_t = torch.from_numpy(seed_np.transpose(0, 4, 1, 2, 3)).to(device)
patches_t = torch.from_numpy(patches_np.transpose(0, 4, 1, 2, 3)).to(device)
```

Then `seed_t` and `patches_t` have shape **`[4, 1, 25, 49, 49]`** and are fed to the 3D ResNet / UNet.

#### 2. Model Output and Loss

```python
logits = model(patches_t, seed_t)
loss = model.compute_loss(logits, labels_t, weights_t)
```

* **Output (`logits`)**: Same shape (e.g. `[4, 1, 25, 49, 49]`), the network’s **current belief** in logit space.
* **Loss**: `compute_loss` uses `F.binary_cross_entropy_with_logits`: Sigmoid on `logits`, compare with soft labels, multiply by `weights_t`, then `.mean()`. Then `loss.backward()` and `optim.step()` complete one training step.

---

### Part 4: The Seed Update Loop and How It Drives the Policy

The model has learned, but if that knowledge is not fed back to the waiting generator, the next move has no basis.

#### 1. Format Conversion Back

```python
updated_seed_np = logits.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
```

We pull the latest logits from GPU to CPU and transpose back to **`[4, 25, 49, 49, 1]` (NDHWC)**.

#### 2. In-Place Overwrite via References

Call `batch_it.update_seeds(updated_seed_np)`. Inside `BatchExampleIter.update_seeds`:

```python
self._seeds[i][:, dz//2:-(dz-dz//2), ...] = batched_seeds[i, ...]
```

`self._seeds` was saved in `__next__`: they are **NumPy view references** into the background canvases. This line **overwrites** those regions with the GPU’s latest output.

#### 3. Closing the Loop

After this overwrite, the background canvas is physically updated (e.g. high probability on the right). On the next `next(batch_it)`, the `fixed_offsets` generator wakes, checks the next direction (e.g. right): `seed[:, right_center, ...]`. **Because we just wrote high probability there, `_eval_move` returns `valid_move = True`.** The generator yields a right move and crops a new patch centered to the right.

**The data loop is complete:** the model predicts from the current canvas, the system writes that prediction back onto the canvas, and the generator uses the updated canvas to decide the next move—simulating how a neuron grows and probes its neighborhood.
