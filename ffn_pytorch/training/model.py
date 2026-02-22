"""FFN model base (PyTorch).

This module defines the core abstractions for Flood-Filling Networks:
- ModelInfo: dataclass for network geometry
- FFNModel: base class for all FFN models (nn.Module)

Original TF used tf.placeholder and session.run() for forward;
PyTorch uses standard nn.Module forward().

Tensor layout:
- PyTorch models use BCZYX (Batch, Channel, Z, Y, X)
- ModelInfo size arrays use XYZ order (same as original TF)
- Training/inference code converts BZYXC ↔ BCZYX

Reference: https://arxiv.org/abs/1611.00421
"""

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelInfo:
    """Geometry parameters for an FFN model.

    All size arrays are (x, y, z) order, matching original TF.
    For 2D models, z is ignored.

    Attributes:
        deltas: FOV step size (x, y, z). E.g. [8, 8, 8] = 8 voxels per direction.
        pred_mask_size: Output prediction mask size (x, y, z). Usually same as fov_size.
        input_seed_size: Input seed mask size (x, y, z). May be larger than pred_mask_size.
        input_image_size: Input image patch size (x, y, z). May be larger than pred_mask_size.
        additive: If True, network output is additive update to seed; else direct logit. Default False.
    """
    deltas: np.ndarray
    pred_mask_size: np.ndarray
    input_seed_size: np.ndarray
    input_image_size: np.ndarray
    additive: bool = False


class FFNModel(nn.Module):
    """Base class for Flood-Filling Network models.

    All FFN implementations (e.g. ConvStack3DFFNModel) must subclass this.

    Subclasses must:
    1. Set class attribute `dim` (2 or 3)
    2. Implement `forward()`

    Inputs:
    - input_patches: image patch [B, 1, Z, Y, X]
    - input_seed: current seed mask [B, 1, Z, Y, X] (logit space)

    Output: updated seed logit [B, 1, Z, Y, X].

    Attributes:
        info: ModelInfo instance (geometry)
        dim: spatial dimension (2 or 3), set by subclass
        shifts: list of all FOV move offsets (26-neighborhood in 3D)
    """

    info: ModelInfo
    dim: int = None

    def __init__(self, info: ModelInfo, batch_size=None):
        super().__init__()
        assert self.dim is not None

        self.info = info
        self.batch_size = batch_size

        # Generate all FOV move directions. For deltas=[8,8,8]: 26 directions (exclude origin).
        self.shifts = []
        for dx in (-self.info.deltas[0], 0, self.info.deltas[0]):
            for dy in (-self.info.deltas[1], 0, self.info.deltas[1]):
                for dz in (-self.info.deltas[2], 0, self.info.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))

    def forward(self, input_patches: torch.Tensor,
                input_seed: torch.Tensor) -> torch.Tensor:
        """Forward pass. Subclasses must implement.

        Args:
            input_patches: image patch [B, 1, Z, Y, X], float32
            input_seed: seed mask [B, 1, Z, Y, X], logit space

        Returns:
            Updated seed logit [B, 1, Z, Y, X]
        """
        raise NotImplementedError(
            'forward() needs to be defined by a subclass.')

    def update_seed(self, input_seed: torch.Tensor,
                    update: torch.Tensor) -> torch.Tensor:
        """Add network logit update to input seed.

        When pred_mask_size < input_seed_size, the network predicts only the center;
        zero-pad update to match input_seed size.

        Args:
            input_seed: original seed [B, 1, Z, Y, X]
            update: network logit update [B, 1, Z', Y', X'], Z'×Y'×X' = pred_mask_size

        Returns:
            Updated seed [B, 1, Z, Y, X]
        """
        # Size difference between input_seed and pred_mask per dimension
        dx = self.info.input_seed_size[0] - self.info.pred_mask_size[0]
        dy = self.info.input_seed_size[1] - self.info.pred_mask_size[1]
        dz = self.info.input_seed_size[2] - self.info.pred_mask_size[2]

        if dx == 0 and dy == 0 and dz == 0:
            seed = input_seed + update
        else:
            # Zero-pad update to input_seed size. F.pad: (x_left, x_right, y_left, y_right, ...)
            seed = input_seed + F.pad(
                update,
                [dx // 2, dx - dx // 2,
                 dy // 2, dy - dy // 2,
                 dz // 2, dz - dz // 2,
                 0, 0,   # channel
                 0, 0])  # batch
        return seed

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                     loss_weights: torch.Tensor) -> torch.Tensor:
        """Weighted sigmoid binary cross-entropy loss.

        Args:
            logits: network prediction [B, 1, Z, Y, X] (pre-sigmoid)
            labels: target [B, 1, Z, Y, X] (soft labels 0.05–0.95)
            loss_weights: per-voxel weights [B, 1, Z, Y, X]

        Returns:
            Scalar loss (weighted mean over voxels)
        """
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')
        pixel_loss = pixel_loss * loss_weights
        loss = pixel_loss.mean()
        return loss
