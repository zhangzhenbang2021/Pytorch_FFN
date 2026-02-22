"""FFN field-of-view (FOV) movement policy (PyTorch).

This module defines how the FOV moves in 3D during FFN inference.

Idea: After predicting at the current FOV position, FFN must decide where to move next.
The movement policy evaluates the prediction probability map to pick the most "foreground"
neighboring positions.

Components:

1. get_scored_move_offsets(): score all possible FOV move directions
   - Compute mean probability on 6 faces (±z, ±y, ±x)
   - Return (score, offset) list above threshold

2. FaceMaxMovementPolicy: standard movement policy
   - Maintains a priority queue (scored_coords)
   - Picks highest-scoring unvisited position
   - Already-segmented regions are marked non-movable

3. MovementRestrictor: position restrictor
   - Excludes regions via a boolean mask
   - seed_mask: restrict seed positions
   - shift_mask: restriction based on shift

4. get_policy_fn(): factory to build movement policy from config

This module is framework-agnostic (pure NumPy/SciPy), no PyTorch or TensorFlow.
"""

from collections import deque
import json
from typing import Optional
import weakref

import numpy as np
from scipy.special import logit

from ..training import model as ffn_model
from ..training.import_util import import_symbol


def get_scored_move_offsets(deltas, prob_map, threshold=0.9):
    """Looks for potential moves for a FFN.

    Args:
        deltas: (z, y, x) tuple of base move offsets
        prob_map: current probability map as a (z, y, x) numpy array
        threshold: minimum score for a valid move

    Yields:
        (score, (z, y, x) position offset) tuples
    """
    center = np.array(prob_map.shape) // 2
    assert center.size == 3
    subvol_sel = [slice(c - dx, c + dx + 1)
                  for c, dx in zip(center, deltas)]

    done = set()
    for axis, axis_delta in enumerate(deltas):
        if axis_delta == 0:
            continue
        for axis_offset in (-axis_delta, axis_delta):
            face_sel = subvol_sel[:]
            face_sel[axis] = axis_offset + center[axis]
            face_prob = prob_map[tuple(face_sel)]
            shape = face_prob.shape

            face_pos = np.unravel_index(face_prob.argmax(), shape)
            score = face_prob[face_pos]

            if score < threshold:
                continue

            relative_pos = [face_pos[0] - shape[0] // 2,
                            face_pos[1] - shape[1] // 2]
            relative_pos.insert(axis, axis_offset)
            ret = (score, tuple(relative_pos))

            if ret not in done:
                done.add(ret)
                yield ret


class BaseMovementPolicy:
    """Base class for movement policy queues."""

    def __init__(self, canvas, scored_coords, deltas):
        self.canvas = weakref.proxy(canvas)
        self.scored_coords = scored_coords
        self.deltas = np.array(deltas)

    def __len__(self):
        return len(self.scored_coords)

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration()

    def next(self):
        return self.__next__()

    def append(self, item):
        self.scored_coords.append(item)

    def update(self, prob_map, position):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def restore_state(self, state):
        raise NotImplementedError()

    def reset_state(self, start_pos):
        raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
    """Selects candidates from maxima on prediction cuboid faces."""

    def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.9):
        self.done_rounded_coords = set()
        self.score_threshold = score_threshold
        self._start_pos = None
        super().__init__(canvas, deque([]), deltas)

    def reset_state(self, start_pos):
        self.scored_coords = deque([])
        self.done_rounded_coords = set()
        self._start_pos = start_pos

    def get_state(self):
        return [(self.scored_coords, self.done_rounded_coords, self._start_pos)]

    def restore_state(self, state):
        self.scored_coords, self.done_rounded_coords, self._start_pos = state[0]

    def __next__(self):
        while self.scored_coords:
            _, coord = self.scored_coords.popleft()
            coord = tuple(coord)
            if self.quantize_pos(coord) in self.done_rounded_coords:
                continue
            if self.canvas.is_valid_pos(coord):
                break
        else:
            raise StopIteration()
        return tuple(coord)

    def quantize_pos(self, pos):
        rel_pos = np.array(pos) - self._start_pos
        coord = (rel_pos + self.deltas // 2) // np.maximum(self.deltas, 1)
        return tuple(coord)

    def update(self, prob_map, position):
        qpos = self.quantize_pos(position)
        self.done_rounded_coords.add(qpos)

        scored_coords = get_scored_move_offsets(
            self.deltas, prob_map, threshold=self.score_threshold)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
            coord = [rel_coord[i] + position[i] for i in range(3)]
            self.scored_coords.append((score, coord))


def get_policy_fn(request, model_info: ffn_model.ModelInfo):
    """Returns a policy class based on the InferenceRequest proto."""
    if request.movement_policy_name:
        movement_policy_class = globals().get(request.movement_policy_name, None)
        if movement_policy_class is None:
            movement_policy_class = import_symbol(request.movement_policy_name)
    else:
        movement_policy_class = FaceMaxMovementPolicy

    if request.movement_policy_args:
        kwargs = json.loads(request.movement_policy_args)
    else:
        kwargs = {}
    if 'deltas' not in kwargs:
        kwargs['deltas'] = model_info.deltas[::-1]
    if 'score_threshold' not in kwargs:
        kwargs['score_threshold'] = logit(request.inference_options.move_threshold)

    return lambda canvas: movement_policy_class(canvas, **kwargs)


class MovementRestrictor:
    """Restricts the movement of the FFN FoV."""

    def __init__(self, mask=None, shift_mask=None, shift_mask_fov=None,
                 shift_mask_threshold=4, shift_mask_scale=1, seed_mask=None):
        self.mask = mask
        self.seed_mask = seed_mask
        self._shift_mask_scale = shift_mask_scale
        self.shift_mask = None

        if shift_mask is not None:
            self.shift_mask = (
                np.max(np.abs(shift_mask), axis=0) >= shift_mask_threshold)
            assert shift_mask_fov is not None
            self._shift_mask_fov_pre_offset = shift_mask_fov.start[::-1]
            self._shift_mask_fov_post_offset = shift_mask_fov.end[::-1] - 1

    def is_valid_seed(self, pos):
        if self.seed_mask is not None and self.seed_mask[pos]:
            return False
        return True

    def is_valid_pos(self, pos):
        if self.mask is not None and self.mask[pos]:
            return False

        if self.shift_mask is not None:
            np_pos = np.array(pos)
            fov_low = np.maximum(np_pos + self._shift_mask_fov_pre_offset, 0)
            fov_high = np_pos + self._shift_mask_fov_post_offset
            start = fov_low // self._shift_mask_scale
            end = fov_high // self._shift_mask_scale

            if np.any(self.shift_mask[
                    fov_low[0]:(fov_high[0] + 1),
                    start[1]:(end[1] + 1),
                    start[2]:(end[2] + 1)]):
                return False
        return True
