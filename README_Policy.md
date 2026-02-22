In FFN logic, the network must not only predict the current shape but also decide where to "step" next to continue prediction. These three strategies represent different stages from "staying in place" to "industrial-grade automatic tracking."

---

### 1. Core Evaluator: `_eval_move`

Before discussing specific strategies, one must understand this "referee" that all strategies call. Its job is to judge whether a move is valid.

* **`valid_move` (network self-assessment)**: Checks the current **prediction canvas (seed)**. If the predicted probability at the target point ≥ threshold, the network considers that location "passable" and allows the move.
* **`wanted_move` (ground-truth view)**: Checks the **true labels**. If the target point in the ground truth actually belongs to that neuron, the "path is correct."
* **Training meaning**: The difference between these two values determines Precision and Recall. If `valid` is true but `wanted` is false, an "over-segmentation (merge)" has occurred.

---

### 2. Strategy One: `no_offsets` (Stay in Place)

This is the simplest strategy, typically used for debugging or training on very small blocks.

* **Logic**: It always returns only `(0, 0, 0)`.
* **Behavior**: The network runs one inference at the current position, records the result, and then stops. It produces no displacement and does not explore the surrounding space.

---

### 3. Strategy Two: `fixed_offsets` (Fixed-Step Training)

This is the **most commonly used strategy during FFN training**.

* **Logic**: It checks a preset list (`fov_shifts`, e.g. 6 directions: up, down, left, right, front, back) in order.
* **Flow**:
1. First check the origin `(0, 0, 0)`.
2. Iterate over each offset in the list (e.g. shift right by 8 pixels).
3. **Conditional trigger**: Only when `_eval_move` returns `valid_move` true (i.e. the network predicts something in that direction) does it `yield` that offset.

* **Purpose**: During training, force the network to try preset moves. Because the training block size is limited, fixed steps ensure the network learns how to "take the next step" within a controlled range.

---

### 4. Strategy Three: `max_pred_offsets` (Greedy Search Inference)

This is the **core algorithm used in the FFN inference phase** and the secret to tracking long-range neurons.

* **Core mechanism**: **Breadth-first search (BFS) + greedy score sorting**.
* **Flow**:
1. **Queue management**: Maintain a `queue` of coordinates to explore and a `done` set to avoid revisiting.
2. **Quantized alignment (`quantized_offset`)**: Map continuous coordinates onto a grid so tiny offsets do not cause infinite loops.
3. **Dynamic exploration**:
* Pop a point from the queue and check if it is out of bounds (`max_radius`).
* Call `_eval_move`. If the prediction does not meet the bar (`not valid`), drop that branch.

4. **Real-time pathfinding (`get_scored_move_offsets`)**:
* If prediction succeeds at the current position, it scans all potential move directions within the current field of view.
* **Greedy sort**: Sort candidate directions by predicted probability (highest first, `reverse=True`).
* **Enqueue**: Add the top-scoring directions to `queue`.

* **Characteristics**:
* **Dynamic**: It does not use a fixed set of 6 directions; it goes where the prediction is confident.
* **Variable length**: As long as the network keeps predicting coherent structure, the `while queue` loop can run many times and track long neurons spanning multiple fields of view.

---

### Summary Comparison

| Strategy | Use case | Move behavior | Termination |
| --- | --- | --- | --- |
| **`no_offsets`** | Debugging | Origin only | Ends after one step |
| **`fixed_offsets`** | **Training** | Preset 6–10 fixed directions | Ends when list is exhausted |
| **`max_pred_offsets`** | **Inference / validation** | Dynamic queue, sorted by prediction score | Queue empty or out of bounds (no path) |

**In one sentence: During training (Fixed), we teach the network how to look at neighbors; during inference (Max Pred), we let the network find the path itself from what it sees.**
