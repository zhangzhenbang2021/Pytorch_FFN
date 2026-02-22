"""FFN training metric tracker (PyTorch).

This module tracks evaluation metrics during training:

1. FOV move quality:
   - CORRECT: move direction correct and executed
   - MISSED: should move but seed value too low (skipped)
   - SPURIOUS: should not move but seed value high (wrong execution)

2. Segmentation quality (per patch):
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1: 2 * P * R / (P + R)
   - Accuracy, specificity

3. Loss: cumulative average of weighted cross-entropy loss

4. Visualization: XY/XZ/YZ ortho-plane comparison (label vs prediction vs weight)

Original TF used TFSyncVariable and tf.summary; PyTorch version uses
pure NumPy accumulation + manual SummaryWriter logging.
"""

import collections
import enum
import io
from typing import Optional, Sequence

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from scipy import special

from . import mask


class MoveType(enum.IntEnum):
    CORRECT = 0
    MISSED = 1
    SPURIOUS = 2


class VoxelType(enum.IntEnum):
    TOTAL = 0
    MASKED = 1


class PredictionType(enum.IntEnum):
    TP = 0
    TN = 1
    FP = 2
    FN = 3


class FovStat(enum.IntEnum):
    TOTAL_VOXELS = 0
    MASKED_VOXELS = 1
    WEIGHTS_SUM = 2


class EvalTracker:
    """Tracks eval results over multiple training steps."""

    def __init__(self, eval_shape: list, shifts: Sequence[tuple]):
        self.eval_threshold = special.logit(0.9)
        self._eval_shape = eval_shape
        self._patch_count = 0

        self._init_counters(shifts)
        self.reset()

    def _init_counters(self, fov_shifts):
        self.moves = np.zeros(3, dtype=np.int64)
        self.loss_val = np.zeros(1, dtype=np.float32)
        self.num_voxels = np.zeros(2, dtype=np.int64)
        self.num_patches = np.zeros(1, dtype=np.int64)
        self.prediction_counts = np.zeros(4, dtype=np.int64)
        self.fov_stats = np.zeros(3, dtype=np.float32)

        radii = set([int(np.linalg.norm(s)) for s in fov_shifts])
        radii.add(0)
        self.moves_by_r = {}
        for r in radii:
            self.moves_by_r[r] = np.zeros(3, dtype=np.int64)

    def reset(self):
        """Resets status of the tracker."""
        self.images_xy = collections.deque(maxlen=16)
        self.images_xz = collections.deque(maxlen=16)
        self.images_yz = collections.deque(maxlen=16)

        self.moves[:] = 0
        self.loss_val[:] = 0
        self.num_voxels[:] = 0
        self.num_patches[:] = 0
        self.prediction_counts[:] = 0
        self.fov_stats[:] = 0
        for r in self.moves_by_r:
            self.moves_by_r[r][:] = 0

    def track_weights(self, weights: np.ndarray):
        self.fov_stats[FovStat.TOTAL_VOXELS] += weights.size
        self.fov_stats[FovStat.MASKED_VOXELS] += np.sum(weights == 0.0)
        self.fov_stats[FovStat.WEIGHTS_SUM] += np.sum(weights)

    def record_move(self, wanted: bool, executed: bool,
                    offset_xyz: Sequence[int]):
        """Records an FFN FOV move."""
        r = int(np.linalg.norm(offset_xyz))
        assert r in self.moves_by_r, f'{r} not in {list(self.moves_by_r.keys())}'

        if wanted:
            if executed:
                self.moves[MoveType.CORRECT] += 1
                self.moves_by_r[r][MoveType.CORRECT] += 1
            else:
                self.moves[MoveType.MISSED] += 1
                self.moves_by_r[r][MoveType.MISSED] += 1
        elif executed:
            self.moves[MoveType.SPURIOUS] += 1
            self.moves_by_r[r][MoveType.SPURIOUS] += 1

    def add_patch(self, labels: np.ndarray, predicted: np.ndarray,
                  weights: np.ndarray, coord: Optional[np.ndarray] = None,
                  image_summaries: bool = True,
                  volume_name: Optional[str] = None):
        """Evaluates single-object segmentation quality."""
        predicted = mask.crop_and_pad(predicted, (0, 0, 0), self._eval_shape)
        weights = mask.crop_and_pad(weights, (0, 0, 0), self._eval_shape)
        labels = mask.crop_and_pad(labels, (0, 0, 0), self._eval_shape)

        sig_pred = special.expit(predicted)
        loss = np.mean(weights * (-labels * np.log(sig_pred + 1e-7)
                                  - (1 - labels) * np.log(1 - sig_pred + 1e-7)))

        self.loss_val[:] += loss
        self.num_voxels[VoxelType.TOTAL] += labels.size
        self.num_voxels[VoxelType.MASKED] += np.sum(weights == 0.0)

        pred_mask = predicted >= self.eval_threshold
        true_mask = labels > 0.5
        pred_bg = np.logical_not(pred_mask)
        true_bg = np.logical_not(true_mask)

        self.prediction_counts[PredictionType.TP] += np.sum(pred_mask & true_mask)
        self.prediction_counts[PredictionType.TN] += np.sum(pred_bg & true_bg)
        self.prediction_counts[PredictionType.FP] += np.sum(pred_mask & true_bg)
        self.prediction_counts[PredictionType.FN] += np.sum(pred_bg & true_mask)
        self.num_patches[:] += 1

        if image_summaries:
            vis_pred = special.expit(predicted)
            self.images_xy.append(
                self._slice_image(coord, labels, vis_pred, weights, 0, volume_name))
            self.images_xz.append(
                self._slice_image(coord, labels, vis_pred, weights, 1, volume_name))
            self.images_yz.append(
                self._slice_image(coord, labels, vis_pred, weights, 2, volume_name))

    def _slice_image(self, coord, labels, predicted, weights,
                     slice_axis, volume_name=None):
        """Builds a visualization image showing label vs prediction."""
        zyx = list(labels.shape[1:-1])
        selector = [0, slice(None), slice(None), slice(None), 0]
        selector[slice_axis + 1] = zyx[slice_axis] // 2
        selector = tuple(selector)

        del zyx[slice_axis]
        h, w = zyx

        labels_slice = (labels[selector] * 255).astype(np.uint8)
        pred_slice = (predicted[selector] * 255).astype(np.uint8)
        weights_slice = (weights[selector] * 255).astype(np.uint8)

        combined = np.concatenate([labels_slice, pred_slice, weights_slice],
                                  axis=1)
        combined = np.repeat(combined[..., np.newaxis], 3, axis=2)

        im = PIL.Image.fromarray(combined, 'RGB')
        draw = PIL.ImageDraw.Draw(im)
        if coord is not None:
            x, y, z = np.array(coord).squeeze()
            text = f'{x},{y},{z}'
            if volume_name is not None:
                if isinstance(volume_name, (list, tuple, np.ndarray)):
                    volume_name = volume_name[0]
                if isinstance(volume_name, bytes):
                    volume_name = volume_name.decode('utf-8')
                text += f'\n{volume_name}'
            try:
                font = PIL.ImageFont.load_default()
            except (IOError, ValueError):
                font = None
            draw.text((1, 1), text, fill='rgb(255,64,64)', font=font)

        buf = io.BytesIO()
        im.save(buf, 'PNG')
        return {'image': buf.getvalue(), 'height': h, 'width': w * 3}

    def get_summaries(self):
        """Returns a dict of summary metrics for tensorboard logging."""
        if not self.num_voxels[VoxelType.TOTAL]:
            return {}

        summaries = {}
        total_moves = sum(self.moves)

        if total_moves > 0:
            for mt in MoveType:
                summaries[f'moves/all/{mt.name.lower()}'] = (
                    self.moves[mt] / total_moves)

        summaries['moves/total'] = total_moves

        if self.fov_stats[FovStat.TOTAL_VOXELS] > 0:
            summaries['fov/masked_voxel_fraction'] = (
                self.fov_stats[FovStat.MASKED_VOXELS] /
                self.fov_stats[FovStat.TOTAL_VOXELS])
            summaries['fov/average_weight'] = (
                self.fov_stats[FovStat.WEIGHTS_SUM] /
                self.fov_stats[FovStat.TOTAL_VOXELS])

        if self.num_voxels[VoxelType.TOTAL] > 0:
            summaries['masked_voxel_fraction'] = (
                self.num_voxels[VoxelType.MASKED] /
                self.num_voxels[VoxelType.TOTAL])

        if self.num_patches[0] > 0:
            summaries['eval/patch_loss'] = (
                self.loss_val[0] / self.num_patches[0])
            summaries['eval/patches'] = self.num_patches[0]

        tp = self.prediction_counts[PredictionType.TP]
        fp = self.prediction_counts[PredictionType.FP]
        tn = self.prediction_counts[PredictionType.TN]
        fn = self.prediction_counts[PredictionType.FN]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision > 0 or recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        summaries['eval/all/accuracy'] = (tp + tn) / max(tp + tn + fp + fn, 1)
        summaries['eval/all/precision'] = precision
        summaries['eval/all/recall'] = recall
        summaries['eval/all/specificity'] = tn / max(tn + fp, 1)
        summaries['eval/all/f1'] = f1

        for r, r_moves in self.moves_by_r.items():
            total_r = sum(r_moves)
            if total_r > 0:
                summaries[f'moves/r={r}/correct'] = (
                    r_moves[MoveType.CORRECT] / total_r)
                summaries[f'moves/r={r}/spurious'] = (
                    r_moves[MoveType.SPURIOUS] / total_r)
                summaries[f'moves/r={r}/missed'] = (
                    r_moves[MoveType.MISSED] / total_r)
            summaries[f'moves/r={r}/total'] = total_r

        return summaries
