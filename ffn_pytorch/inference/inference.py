"""FFN inference core module (PyTorch).

This module contains the Canvas class—the core data structure for FFN inference.

Canvas maintains full inference state for a subvolume:
- seed: current working mask (logit space), "probability each voxel belongs to current object"
- segmentation: completed segmentation (integer IDs)
- seg_prob: optional probability map

Core inference loop (segment_all → segment_at → update_at → predict):

1. segment_all(): outer loop over all seed positions
   ├─ seed_policy generates candidate start positions
   ├─ validate position (bounds, already segmented, threshold)
   ├─ call segment_at() to segment one object
   └─ evaluate result (size, quality), assign segment ID

2. segment_at(): middle loop, iteratively extend one object
   ├─ init_seed(): place initial seed at start position
   ├─ iterate over FOV positions from movement_policy
   ├─ call update_at() to update prediction
   └─ call movement_policy.update() to update move queue

3. update_at(): single-step inference
   ├─ crop current mask for FOV from seed
   ├─ call predict() → executor → network inference
   ├─ apply disco_seed_threshold (prevent disconnected prediction reversal)
   └─ write prediction back to seed

4. predict(): wrap network call
   ├─ get image at FOV position
   └─ submit inference request via executor client

All spatial coordinates use (z, y, x) order (matches NumPy array axes).
ModelInfo deltas etc. use (x, y, z) order; use [::-1] to convert.

Differences from original TF version:
- Replaced tensorflow.io.gfile with standard Python os
- Canvas is framework-agnostic (pure NumPy); PyTorch only via executor
"""

from io import BytesIO
import logging
import os
import threading
import time

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.special import expit
from scipy.special import logit

from ..training import model as ffn_model
from . import executor
from . import movement
from . import seed
from . import storage
from .inference_utils import Counters
from .inference_utils import TimedIter
from .inference_utils import timer_counter

try:
    from ffn.inference import inference_pb2
except (ImportError, TypeError):
    inference_pb2 = None

try:
    from ffn.utils import ortho_plane_visualization
except (ImportError, TypeError):
    ortho_plane_visualization = None


MSEC_IN_SEC = 1000
MAX_SELF_CONSISTENT_ITERS = 32

from typing import Tuple
Tuple3i = Tuple[int, int, int]


# Visualization.
# ---------------------------------------------------------------------------

class DynamicImage:
    def UpdateFromPIL(self, new_img):
        from IPython import display
        display.clear_output(wait=True)
        image = BytesIO()
        new_img.save(image, format='png')
        display.display(display.Image(image.getvalue()))


def _cmap_rgb1(drw):
    """Default color palette from gnuplot."""
    r = np.sqrt(drw)
    g = np.power(drw, 3)
    b = np.sin(drw * np.pi)
    return (np.dstack([r, g, b]) * 250.0).astype(np.uint8)


def visualize_state(seed_logits, pos: Tuple3i, movement_policy, dynimage):
    """Visualizes the inference state."""
    from PIL import Image

    if ortho_plane_visualization is None:
        return

    planes = ortho_plane_visualization.cut_ortho_planes(
        seed_logits, center=pos, cross_hair=True)
    to_vis = ortho_plane_visualization.concat_ortho_planes(planes)

    if isinstance(movement_policy.scored_coords, np.ndarray):
        scores = movement_policy.scored_coords
        zf, yf, xf = movement_policy.deltas
        zz, yy, xx = scores.shape
        zs, ys, xs = scores.strides
        new_sh = (zz, zf, yy, yf, xx, xf)
        new_st = (zs, 0, ys, 0, xs, 0)
        scores_up = as_strided(scores, new_sh, new_st)
        scores_up = scores_up.reshape((zz * zf, yy * yf, xx * xf))
        cut = (np.array(scores_up.shape) - np.array(seed_logits.shape)) // 2
        sh = seed_logits.shape
        scores_up = scores_up[
            cut[0]:cut[0] + sh[0],
            cut[1]:cut[1] + sh[1],
            cut[2]:cut[2] + sh[2],
        ]
        grid_planes = ortho_plane_visualization.cut_ortho_planes(
            scores_up, center=pos, cross_hair=True)
        grid_view = ortho_plane_visualization.concat_ortho_planes(grid_planes)
        grid_view *= 4
        to_vis = np.concatenate((to_vis, grid_view), axis=1)

    val = _cmap_rgb1(expit(to_vis))
    y, x = pos[1:]

    val[(y - 1):(y + 2), (x - 1):(x + 2), 0] = 255
    val[(y - 1):(y + 2), (x - 1):(x + 2), 1:] = 0

    vis = Image.fromarray(val)
    dynimage.UpdateFromPIL(vis)


class Canvas:
    """Tracks state of the inference progress and results within a subvolume."""

    io_lock = threading.Lock()

    def __init__(
        self,
        model_info: ffn_model.ModelInfo,
        exec_client: executor.ExecutorClient,
        image,
        options,
        voxel_size_zyx: Tuple[int, int, int] = (1, 1, 1),
        counters=None,
        restrictor=None,
        movement_policy_fn=None,
        keep_history=False,
        checkpoint_path=None,
        checkpoint_interval_sec=0,
        corner_zyx=None,
        storage_cls=storage.NumpyArray,
        keep_probability_maps=False,
    ):
        self.image = image
        self._exec_client = exec_client
        self._exec_client_id = None
        self.voxel_size_zyx = voxel_size_zyx

        if inference_pb2 is not None:
            self.options = inference_pb2.InferenceOptions()
            self.options.CopyFrom(options)
        else:
            self.options = options

        for attr in (
            'init_activation',
            'pad_value',
            'move_threshold',
            'segment_threshold',
        ):
            setattr(self.options, attr, logit(getattr(self.options, attr)))

        self.counters = counters if counters is not None else Counters()
        self.checkpoint_interval_sec = checkpoint_interval_sec
        self.checkpoint_path = checkpoint_path
        self.checkpoint_last = time.time()

        self._keep_history = keep_history
        self.corner_zyx = corner_zyx
        self.shape = image.shape

        if restrictor is None:
            self.restrictor = movement.MovementRestrictor()
        else:
            self.restrictor = restrictor

        self._pred_size = np.array(model_info.pred_mask_size[::-1])
        self._input_seed_size = np.array(model_info.input_seed_size[::-1])
        self._input_image_size = np.array(model_info.input_image_size[::-1])
        self.margin = self._input_image_size // 2

        self._pred_delta = (self._input_seed_size - self._pred_size) // 2
        assert np.all(self._pred_delta >= 0)

        self.seed = storage_cls(
            shape=self.shape, dtype=np.float32, default_value=np.nan)
        self.segmentation = storage_cls(shape=self.shape, dtype=np.int32)
        self.keep_probability_maps = keep_probability_maps
        if keep_probability_maps:
            self.seg_prob = storage_cls(shape=self.shape, dtype=np.uint8)
        else:
            self.seg_prob = None

        self.global_to_local_ids = {}
        self.local_to_global_ids = {}

        self.seed_policy = None
        self._seed_policy_state = None

        self._max_id = 0

        self.origins = {}
        self.overlaps = {}

        self.reset_seed_per_segment = True

        if movement_policy_fn is None:
            self.movement_policy = movement.FaceMaxMovementPolicy(
                self,
                deltas=model_info.deltas[::-1],
                score_threshold=self.options.move_threshold,
            )
        else:
            self.movement_policy = movement_policy_fn(self)

        self._hosts = []
        self.reset_state((0, 0, 0))
        self.t_last_predict = None
        self.log_info(
            'Constructed canvas with corner %s (zyx) and shape %s',
            self.corner_zyx,
            self.shape,
        )

    def _register_client(self):
        if self._exec_client_id is None:
            self._exec_client_id = self._exec_client.start()
            logging.info('Registered as client %d.', self._exec_client_id)

    def _deregister_client(self):
        if self._exec_client_id is not None:
            logging.info('Deregistering client %d', self._exec_client_id)
            self._exec_client.finish()
            self._exec_client_id = None

    def __del__(self):
        self._deregister_client()

    def local_id(self, segment_id: int):
        return self.global_to_local_ids.get(segment_id, segment_id)

    def reset_state(self, start_pos: Tuple3i, reset_extents=True):
        self.movement_policy.reset_state(start_pos)
        self.history = []
        self.history_deleted = []

        if reset_extents:
            self._min_pos = np.array(start_pos)
            self._max_pos = np.array(start_pos)

        self._register_client()

    def is_valid_pos(self, pos: Tuple3i, ignore_move_threshold=False) -> bool:
        if not ignore_move_threshold:
            if self.seed[pos] < self.options.move_threshold:
                self.counters['skip_threshold'].Increment()
                logging.debug('.. seed value below threshold.')
                return False

        np_pos = np.array(pos)
        low = np_pos - self.margin
        high = np_pos + self.margin

        if np.any(low < 0) or np.any(high >= self.shape):
            self.counters['skip_invalid_pos'].Increment()
            logging.debug('.. too close to border: %r', pos)
            return False

        if self.segmentation[pos] > 0:
            self.counters['skip_invalid_pos'].Increment()
            logging.debug('.. segmentation already active: %r', pos)
            return False

        return True

    def _get_image(self, pos: Tuple3i) -> np.ndarray:
        start = np.array(pos) - self.margin
        end = start + self._input_image_size
        img = self.image[tuple(slice(s, e) for s, e in zip(start, end))]
        return img

    def predict(self, pos: Tuple3i, logit_seed: np.ndarray) -> np.ndarray:
        with timer_counter(self.counters, 'predict'):
            with timer_counter(self.counters, 'get-image'):
                img = self._get_image(pos)

            if self.t_last_predict is not None:
                delta_t = time.time() - self.t_last_predict
                self.counters['inference-not-predict-ms'].IncrementBy(
                    delta_t * MSEC_IN_SEC)
            extra_fetches = ['logits']

            with timer_counter(self.counters, 'inference'):
                fetches = self._exec_client.predict(logit_seed, img, extra_fetches)

            self.t_last_predict = time.time()

        logits = fetches.pop('logits')
        return logits[..., 0]

    def update_at(self, pos: Tuple3i) -> np.ndarray:
        with timer_counter(self.counters, 'update_at'):
            off = self._input_seed_size // 2

            start = np.array(pos) - off
            end = start + self._input_seed_size
            logit_seed = np.array(
                self.seed[tuple(slice(s, e) for s, e in zip(start, end))])
            init_prediction = np.isnan(logit_seed)
            logit_seed[init_prediction] = np.float32(self.options.pad_value)

            logits = self.predict(pos, logit_seed)
            start += self._pred_delta
            end = start + self._pred_size
            sel = tuple(slice(s, e) for s, e in zip(start, end))

            if self.options.disco_seed_threshold >= 0:
                th_max = logit(0.5)
                old_seed = self.seed[sel]

                if self._keep_history:
                    self.history_deleted.append(
                        np.sum((old_seed >= logit(0.8)) & (logits < th_max)))

                if (np.mean(logits >= self.options.move_threshold)
                        > self.options.disco_seed_threshold):
                    old_err = np.seterr(invalid='ignore')
                    try:
                        mask = (old_seed < th_max) & (logits > old_seed)
                    finally:
                        np.seterr(**old_err)
                    logits[mask] = old_seed[mask]

            self.seed[sel] = logits

        return logits

    def init_seed(self, pos: Tuple3i):
        self.seed.clear()
        self.seed[pos] = self.options.init_activation

    def get_next_segment_id(self) -> int:
        self._max_id += 1
        while self._max_id in self.origins:
            self._max_id += 1
        return self._max_id

    def segment_at(
        self,
        start_pos: Tuple3i,
        dynamic_image=None,
        vis_update_every=10,
        vis_fixed_z=False,
        partial_segment_iters=0,
    ):
        if not partial_segment_iters:
            if self.reset_seed_per_segment:
                self.init_seed(start_pos)
            self.reset_state(start_pos, reset_extents=self.reset_seed_per_segment)

            if not self.movement_policy:
                item = (self.movement_policy.score_threshold * 2, start_pos)
                self.movement_policy.append(item)

        num_iters = partial_segment_iters

        with timer_counter(self.counters, 'segment_at-loop'):
            for pos in self.movement_policy:
                if self.seed[start_pos] < self.options.move_threshold:
                    self.counters['seed_got_too_weak'].Increment()
                    break

                if not self.restrictor.is_valid_pos(pos):
                    self.counters['skip_restriced_pos'].Increment()
                    continue

                pred = self.update_at(pos)
                self._min_pos = np.minimum(self._min_pos, pos)
                self._max_pos = np.maximum(self._max_pos, pos)
                num_iters += 1

                with timer_counter(self.counters, 'movement_policy'):
                    self.movement_policy.update(pred, pos)

                with timer_counter(self.counters, 'segment_at-overhead'):
                    if self._keep_history:
                        self.history.append(pos)

                    if dynamic_image is not None and num_iters % vis_update_every == 0:
                        vis_pos = pos if not vis_fixed_z else (start_pos[0], pos[1], pos[2])
                        visualize_state(
                            self.seed, vis_pos, self.movement_policy, dynamic_image)

                    assert np.all(pred.shape == self._pred_size)
                    self._maybe_save_checkpoint(partial_segment_iters=num_iters)

        return num_iters

    def log_info(self, string: str, *args, **kwargs):
        logging.info('[cl %d] ' + string, self._exec_client_id, *args, **kwargs)

    def segment_all(self, seed_policy=seed.PolicyPeaks, partial_segment_iters=0):
        self.seed_policy = seed_policy(self)
        if self._seed_policy_state is not None:
            self.seed_policy.set_state(self._seed_policy_state)
            self._seed_policy_state = None

        with timer_counter(self.counters, 'segment_all'):
            mbd = self.options.min_boundary_dist
            mbd = np.array([mbd.z, mbd.y, mbd.x])

            for pos in TimedIter(self.seed_policy, self.counters, 'seed-policy'):
                if not (
                    self.is_valid_pos(pos, ignore_move_threshold=True)
                    and self.restrictor.is_valid_pos(pos)
                    and self.restrictor.is_valid_seed(pos)
                ):
                    assert not partial_segment_iters
                    continue

                if not partial_segment_iters:
                    self._maybe_save_checkpoint(partial_segment_iters=0)

                low = np.array(pos) - mbd
                high = np.array(pos) + mbd + 1
                sel = tuple(slice(s, e) for s, e in zip(low, high))
                if np.any(self.segmentation[sel] > 0):
                    logging.debug('Too close to existing segment.')
                    self.segmentation[pos] = -1
                    assert not partial_segment_iters
                    continue

                self.log_info('Starting segmentation at %r (zyx)', pos)

                seg_start = time.time()
                num_iters = self.segment_at(
                    pos, partial_segment_iters=partial_segment_iters)
                partial_segment_iters = 0
                t_seg = time.time() - seg_start

                if num_iters <= 0:
                    self.counters['invalid-other-time-ms'].IncrementBy(
                        t_seg * MSEC_IN_SEC)
                    self.log_info('Failed: num iters was %d', num_iters)
                    continue

                if self.seed[pos] < self.options.move_threshold:
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    self.log_info('Failed: weak seed')
                    self.counters['invalid-weak-time-ms'].IncrementBy(
                        t_seg * MSEC_IN_SEC)
                    continue

                sel = tuple(
                    slice(max(s, 0), e + 1)
                    for s, e in zip(
                        self._min_pos - self._pred_size // 2,
                        self._max_pos + self._pred_size // 2,
                    ))

                mask = self.seed[sel] >= self.options.segment_threshold
                raw_segmented_voxels = np.sum(mask)

                overlapped_ids, counts = np.unique(
                    self.segmentation[sel][mask], return_counts=True)
                valid = overlapped_ids > 0
                overlapped_ids = overlapped_ids[valid]
                counts = counts[valid]

                mask &= self.segmentation[sel] <= 0
                actual_segmented_voxels = np.sum(mask)

                if actual_segmented_voxels < self.options.min_segment_size:
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    self.log_info('Failed: too small: %d', actual_segmented_voxels)
                    self.counters['invalid-small-time-ms'].IncrementBy(
                        t_seg * MSEC_IN_SEC)
                    continue

                self.counters['voxels-segmented'].IncrementBy(actual_segmented_voxels)
                self.counters['voxels-overlapping'].IncrementBy(
                    raw_segmented_voxels - actual_segmented_voxels)

                sid = self.get_next_segment_id()
                self.segmentation[sel][mask] = sid
                if self.keep_probability_maps:
                    self.seg_prob[sel][mask] = storage.quantize_probability(
                        expit(self.seed[sel][mask]))

                self.log_info(
                    'Created supervoxel:%d  seed(zyx):%s  size:%d  iters:%d',
                    self._max_id,
                    pos,
                    actual_segmented_voxels,
                    num_iters,
                )

                self.overlaps[self._max_id] = np.array([overlapped_ids, counts])
                self.origins[self._max_id] = storage.OriginInfo(pos, num_iters, t_seg)
                self.counters['valid-time-ms'].IncrementBy(t_seg * MSEC_IN_SEC)
                self._maybe_save_checkpoint(partial_segment_iters=0)

        self.log_info('Segmentation done.')
        self._deregister_client()

    def init_segmentation_from_volume(
        self, volume, corner, end, align_and_crop=None
    ):
        from . import segmentation as seg_module

        init_seg = volume[
            :, corner[0]:end[0], corner[1]:end[1], corner[2]:end[2]]
        init_seg = init_seg[0, ...]
        self.log_info(
            'Segmentation loaded, shape: %r. Canvas segmentation is %r',
            init_seg.shape,
            self.segmentation.shape,
        )

        init_seg, global_to_local = seg_module.labels.make_contiguous(init_seg)
        self.global_to_local_ids = dict(global_to_local)
        self.local_to_global_ids = {
            v: k for k, v in self.global_to_local_ids.items()}

        if align_and_crop is not None:
            init_seg = align_and_crop(init_seg)
            self.log_info('Segmentation cropped to: %r', init_seg.shape)

        self.segmentation[:] = init_seg
        if self.keep_probability_maps:
            assert self.seg_prob is not None
            self.seg_prob[self.segmentation > 0] = storage.quantize_probability(
                np.array([1.0]))
        self._max_id = int(np.max(self.segmentation))
        self.log_info('Max restored ID is: %d.', self._max_id)

    def restore_checkpoint(self, path: str) -> int:
        self.log_info('Restoring inference checkpoint: %s', path)
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)

            self.segmentation[:] = data['segmentation']
            self.seed[:] = data['seed']
            if self.keep_probability_maps:
                assert self.seg_prob is not None
                self.seg_prob[:] = data['seg_qprob']
            self.history_deleted = list(data['history_deleted'])
            self.history = list(data['history'])
            self.origins = data['origins'].item()
            if 'overlaps' in data:
                self.overlaps = data['overlaps'].item()

            segmented_voxels = np.sum(self.segmentation != 0)
            self.counters['voxels-segmented'].Set(segmented_voxels)
            self._max_id = int(np.max(self.segmentation))
            self._min_pos = data['min_pos']
            self._max_pos = data['max_pos']

            self.movement_policy.restore_state(data['movement_policy'])

            self._seed_policy_state = data['seed_policy_state']
            self.counters.loads(data['counters'].item())

            if 'partial_segment_iters' in data:
                partial_segment_iters = data['partial_segment_iters']
            else:
                partial_segment_iters = 0

            if 'hosts' in data:
                self._hosts = list(data['hosts'])

        self.log_info('Inference checkpoint restored.')
        return partial_segment_iters

    def save_checkpoint(self, path: str, partial_segment_iters: int):
        self.log_info('Saving inference checkpoint to %s.', path)
        with timer_counter(self.counters, 'save_checkpoint'):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with storage.atomic_file(path) as fd:
                seed_policy_state = None
                if self.seed_policy is not None:
                    seed_policy_state = self.seed_policy.get_state(
                        partial_segment_iters > 0)

                aux = {}
                if self.keep_probability_maps:
                    aux['seg_qprob'] = self.seg_prob

                np.savez_compressed(
                    fd,
                    movement_policy=np.asarray(
                        self.movement_policy.get_state(), dtype=object),
                    segmentation=self.segmentation,
                    seed=self.seed,
                    origins=self.origins,
                    overlaps=self.overlaps,
                    min_pos=self._min_pos,
                    max_pos=self._max_pos,
                    history=np.array(self.history),
                    history_deleted=np.array(self.history_deleted),
                    seed_policy_state=np.asarray(seed_policy_state, dtype=object),
                    counters=self.counters.dumps(),
                    partial_segment_iters=partial_segment_iters,
                    hosts=self._hosts,
                    **aux,
                )
        self.log_info('Inference checkpoint saved.')

    def _maybe_save_checkpoint(self, partial_segment_iters=0):
        if self.checkpoint_path is None or self.checkpoint_interval_sec <= 0:
            return

        if time.time() - self.checkpoint_last < self.checkpoint_interval_sec:
            return

        with Canvas.io_lock:
            self.save_checkpoint(
                self.checkpoint_path, partial_segment_iters=partial_segment_iters)
        self.checkpoint_last = time.time()
