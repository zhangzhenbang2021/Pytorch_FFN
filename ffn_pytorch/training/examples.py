"""FFN training sample generator (PyTorch).

This module is the core of FFN training: it simulates FOV movement during
inference to produce training samples.

FFN training is unique in that each sample is not independent but a **multi-step
sequence**. The pipeline mimics inference-time iterative filling:
1. Create an initial seed at the center of the label volume
2. Decide FOV move direction from current seed state and labels
3. At each FOV position, crop (image, seed, label, weight)
4. After the network predicts at that position, update seed state
5. Seed state is passed to the next step, forming a dependency chain

FOV movement policies (get_offsets):
- fixed_offsets: predefined move list (26-neighborhood from model.shifts)
- max_pred_offsets: simulates inference greedy policy (highest-scoring direction)
- no_offsets: no move, train only at center

BatchExampleIter design:
- Thread pool to generate multiple independent training sequences in parallel
- update_seeds() feeds network predictions back into seeds for cross-step state
"""

import collections
from concurrent import futures
import itertools
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import special

from ..inference import movement
from . import mask
from . import model as ffn_model
from . import tracker

GetOffsets = Callable[
    [ffn_model.ModelInfo, np.ndarray, np.ndarray, tracker.EvalTracker],
    Iterable[tuple]]


def get_example(load_example, eval_tracker: tracker.EvalTracker,
                info: ffn_model.ModelInfo, get_offsets: GetOffsets,
                seed_pad: float, seed_shape: tuple):
    """Generates individual training examples.

    Args:
        load_example: callable returning (patches, labels, loss_weights, coord, volname)
        eval_tracker: EvalTracker object
        info: ModelInfo metadata
        get_offsets: callable returning iterable of (x, y, z) offsets
        seed_pad: value for empty seed areas
        seed_shape: z, y, x shape of the seed

    Yields:
        tuple of [1, z, y, x, 1]-shaped arrays for: seed, image, label, weights
    """
    while True:
        ex = load_example()
        full_patches, full_labels, loss_weights, coord, volname = ex

        seed = special.logit(mask.make_seed(seed_shape, 1, pad=seed_pad))

        for off in get_offsets(info, seed, full_labels, eval_tracker):
            predicted = mask.crop_and_pad(seed, off, info.input_seed_size[::-1])
            patches = mask.crop_and_pad(full_patches, off,
                                        info.input_image_size[::-1])
            labels = mask.crop_and_pad(full_labels, off,
                                       info.pred_mask_size[::-1])
            weights = mask.crop_and_pad(loss_weights, off,
                                        info.pred_mask_size[::-1])

            assert predicted.base is seed
            yield predicted, patches, labels, weights

        eval_tracker.add_patch(full_labels, seed, loss_weights, coord,
                               volume_name=volname)


ExampleGenerator = Iterable[tuple]


def _batch_gen(make_example_generator_fn: Callable[[], ExampleGenerator],
               batch_size: int):
    """Generates batches of training examples."""
    example_gens = [make_example_generator_fn() for _ in range(batch_size)]

    with futures.ThreadPoolExecutor(max_workers=batch_size) as tpe:
        while True:
            fs = []
            for gen in example_gens:
                fs.append(tpe.submit(next, gen))
            batch = [f.result() for f in fs]
            yield tuple(zip(*batch))


class BatchExampleIter:
    """Generates batches of training examples."""

    def __init__(self, example_generator_fn, eval_tracker, batch_size, info):
        self._eval_tracker = eval_tracker
        self._batch_generator = _batch_gen(example_generator_fn, batch_size)
        self._seeds = None
        self._info = info

    def __iter__(self):
        return self

    def __next__(self):
        seeds, patches, labels, weights = next(self._batch_generator)
        self._seeds = seeds
        batched_seeds = np.concatenate(seeds)
        batched_weights = np.concatenate(weights)
        self._eval_tracker.track_weights(batched_weights)
        return (batched_seeds, np.concatenate(patches),
                np.concatenate(labels), batched_weights)

    def update_seeds(self, batched_seeds):
        """Distributes updated predictions back to the generator buffers.

        Args:
            batched_seeds: [b, z, y, x, c] ndarray of updated seed values
        """
        assert self._seeds is not None
        batched_seeds = np.asarray(batched_seeds)

        dx = self._info.input_seed_size[0] - self._info.pred_mask_size[0]
        dy = self._info.input_seed_size[1] - self._info.pred_mask_size[1]
        dz = self._info.input_seed_size[2] - self._info.pred_mask_size[2]

        if dz == 0 and dy == 0 and dx == 0:
            for i in range(len(self._seeds)):
                self._seeds[i][:] = batched_seeds[i, ...]
        else:
            for i in range(len(self._seeds)):
                self._seeds[i][:,
                               dz // 2:-(dz - dz // 2),
                               dy // 2:-(dy - dy // 2),
                               dx // 2:-(dx - dx // 2),
                               :] = batched_seeds[i, ...]


def _eval_move(seed, labels, off_xyz, seed_threshold, label_threshold):
    """Evaluates a FOV move."""
    valid_move = seed[:,
                      seed.shape[1] // 2 + off_xyz[2],
                      seed.shape[2] // 2 + off_xyz[1],
                      seed.shape[3] // 2 + off_xyz[0],
                      0] >= seed_threshold
    wanted_move = (
        labels[:,
               labels.shape[1] // 2 + off_xyz[2],
               labels.shape[2] // 2 + off_xyz[1],
               labels.shape[3] // 2 + off_xyz[0],
               0] >= label_threshold)
    return valid_move, wanted_move


def fixed_offsets(info, seed, labels, eval_tracker, threshold,
                  fov_shifts=None):
    """Generates offsets based on a fixed list."""
    del info
    label_threshold = special.expit(threshold)
    for off in itertools.chain([(0, 0, 0)], fov_shifts):
        valid_move, wanted_move = _eval_move(seed, labels, off, threshold,
                                             label_threshold)
        eval_tracker.record_move(wanted_move, valid_move, off)
        if not valid_move:
            continue
        yield off


def no_offsets(info, seed, labels, eval_tracker):
    del info, labels, seed
    eval_tracker.record_move(True, True, (0, 0, 0))
    yield (0, 0, 0)


def max_pred_offsets(info, seed, labels, eval_tracker, threshold,
                     max_radius):
    """Generates offsets with the policy used for inference."""
    queue = collections.deque([(0, 0, 0)])
    done = set()

    label_threshold = special.expit(threshold)
    deltas = np.array(info.deltas)

    while queue:
        offset = np.array(queue.popleft())

        if np.any(np.abs(np.array(offset)) > max_radius):
            continue

        quantized_offset = tuple(
            (offset + deltas / 2) // np.maximum(deltas, 1))

        if quantized_offset in done:
            continue

        valid, wanted = _eval_move(seed, labels, tuple(offset), threshold,
                                   label_threshold)
        eval_tracker.record_move(wanted, valid, (0, 0, 0))

        if not valid or (not wanted and quantized_offset != (0, 0, 0)):
            continue

        done.add(quantized_offset)
        yield tuple(offset)

        curr_seed = mask.crop_and_pad(seed, offset, info.pred_mask_size[::-1])
        todos = sorted(
            movement.get_scored_move_offsets(
                info.deltas[::-1], curr_seed[0, ..., 0],
                threshold=threshold),
            reverse=True)
        queue.extend((x[2] + offset[0], x[1] + offset[1], x[0] + offset[2])
                     for _, x in todos)
