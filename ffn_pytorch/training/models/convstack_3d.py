"""3D conv stack FFN model (PyTorch).

Implements the standard FFN architecture from https://arxiv.org/abs/1611.00421:
- Input: concat(image, seed) [B, 2, Z, Y, X]
- Body: multiple 3D residual conv blocks (constant spatial resolution)
- Output: logit update [B, 1, Z, Y, X], added to input seed for final prediction

Original TF used tf_slim.conv3d + tf_slim.batch_norm;
PyTorch uses nn.Conv3d + nn.ReLU + residual connections.

Structure:
    concat(image, seed) [B, 2, Z, Y, X]
           │
           ▼
    Conv3d(2→f, 3³) + ReLU + Conv3d(f→f, 3³)     ← input_block
           │
           ▼
    ┌─→ ReLU → Conv3d(f→f, 3³) → ReLU → Conv3d(f→f, 3³) ─┐
    │                                                        │
    └──────────────────── + (residual) ←────────────────────┘
           │  × (depth - 1) residual blocks
           ▼
    ReLU → Conv3d(f→1, 1³)                         ← output_conv
           │
           ▼
    update_seed(input_seed, logit_update) → [B, 1, Z, Y, X]
"""

import itertools

import torch
import torch.nn as nn

from .. import model


class ConvStack3DFFNModel(model.FFNModel):
    """3D conv stack FFN model.

    Model has `depth` residual blocks; spatial resolution is constant
    (no down/upsampling). This allows arbitrary FOV size with compute
    proportional to FOV volume.

    Args:
        fov_size: FOV size (x, y, z), e.g. [33, 33, 33]. Also used for
                  pred_mask_size, input_seed_size, input_image_size.
        deltas: FOV step (x, y, z), e.g. [8, 8, 8].
        batch_size: Batch size (informational only).
        depth: Number of residual blocks including initial block. Default 9.
               depth=9 is the paper's standard.
        features: Conv channels. Either:
                  - int: same channels for all layers (default 32)
                  - list[int]: length 2*depth, two values per residual block
    """

    dim = 3  # 3D model

    def __init__(
        self,
        fov_size=None,
        deltas=None,
        batch_size=None,
        depth: int = 9,
        features: int = 32,
        **kwargs
    ):
        # fov_size used for all four size parameters
        info = model.ModelInfo(deltas, fov_size, fov_size, fov_size)
        super().__init__(info, batch_size, **kwargs)
        self.depth = depth

        # Build feature channel list: each residual block needs 2 channel counts (two convs)
        if isinstance(features, int):
            feats = list(itertools.repeat(features, 2 * depth))
        else:
            feats = list(features)

        feat_iter = iter(feats)
        in_channels = 2  # image(1ch) + seed(1ch) concat

        # === Input block ===
        # Conv3d → ReLU → Conv3d
        # Map 2-channel input to features channels
        layers = []
        f0 = next(feat_iter)
        layers.append(nn.Conv3d(in_channels, f0, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        f1 = next(feat_iter)
        layers.append(nn.Conv3d(f0, f1, 3, padding=1))

        self.input_block = nn.Sequential(*layers)

        # === Residual blocks ===
        # Each block: ReLU → Conv3d → ReLU → Conv3d, then add input (residual)
        # inplace=False for first ReLU so input is kept for residual
        self.res_blocks = nn.ModuleList()
        for i in range(1, depth):
            fa = next(feat_iter)
            fb = next(feat_iter)
            block = nn.Sequential(
                nn.ReLU(inplace=False),  # no inplace; input used for residual
                nn.Conv3d(f1 if i == 1 else fb_prev, fa, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(fa, fb, 3, padding=1),
            )
            self.res_blocks.append(block)
            fb_prev = fb

        # === Output conv ===
        # 1×1×1 conv maps features to single-channel logit output
        last_feat = fb_prev if depth > 1 else f1
        self.output_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(last_feat, 1, 1, padding=0),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights: truncated normal (std=0.01), bias zero.

        Matches original TF tf.truncated_normal_initializer(stddev=0.01)
        and tf.zeros_initializer().
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_patches: torch.Tensor,
                input_seed: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_patches: image patch [B, 1, Z, Y, X], normalized float32
            input_seed: seed mask [B, 1, Z, Y, X], logit space

        Returns:
            Updated seed logit [B, 1, Z, Y, X]
        """
        # Concat image and seed as input (2 channels)
        net = torch.cat([input_patches, input_seed], dim=1)  # [B, 2, Z, Y, X]

        # Input block
        net = self.input_block(net)

        # Residual blocks (pre-activation style)
        for res_block in self.res_blocks:
            in_net = net
            net = res_block(net)
            net = net + in_net  # residual

        # 1×1×1 output conv for logit update
        logit_update = self.output_conv(net)

        # Add logit update to input seed
        logit_seed = self.update_seed(input_seed, logit_update)
        return logit_seed
