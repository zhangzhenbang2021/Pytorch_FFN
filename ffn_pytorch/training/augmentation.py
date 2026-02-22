"""Augmentation for serial-section EM (ssEM) data (PyTorch).

This module implements augmentations suited to connectomics data.
ssEM is anisotropic (XY resolution higher than Z), so many augmentations
are applied per Z slice to mimic real degradation.

Augmentations fall into two groups:
1. Geometric:
   - PermuteAndReflect: random axis permutation and flip (global)
   - elastic_warp: elastic deformation (tissue deformation)
   - affine_transform: affine (rotate/scale/shear)
   - misalignment: slice alignment error (shift/translate)

2. Intensity:
   - missing_section: simulate missing slices (constant fill)
   - out_of_focus_section: simulate out-of-focus slices (Gaussian blur)
   - grayscale_perturb: brightness/contrast/gamma

Each augmentation has skip_ratio (probability of skipping), so
training data includes some unaugmented samples.

Original TF PermuteAndReflect used TF graph ops; this version uses
pure NumPy (random choice at construction, applied in __call__).
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from skimage.transform import AffineTransform, warp


class PermuteAndReflect:
    """Performs random permutation and reflection of axes (pure NumPy).

    Unlike the TF version which builds graph ops, this generates random
    decisions at construction time and applies them via __call__.
    """

    def __init__(self, rank, permutable_axes, reflectable_axes):
        self.rank = rank
        self.permutable_axes = np.array(permutable_axes, dtype=np.int32)
        self.reflectable_axes = np.array(reflectable_axes, dtype=np.int32)

        self.reflect_decisions = None
        self.full_permutation = None

        if self.reflectable_axes.size > 0:
            self.reflect_decisions = np.random.rand(
                len(self.reflectable_axes)) > 0.5

        if self.permutable_axes.size > 0:
            perm = np.random.permutation(self.permutable_axes)
            full_perm = np.arange(rank, dtype=np.int32)
            for i, d in enumerate(self.permutable_axes):
                full_perm[d] = perm[i]
            self.full_permutation = full_perm

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.full_permutation is not None:
            x = np.transpose(x, self.full_permutation)
        if self.reflect_decisions is not None:
            axes_to_flip = self.reflectable_axes[self.reflect_decisions]
            for ax in axes_to_flip:
                x = np.flip(x, axis=ax)
            x = np.ascontiguousarray(x)
        return x


def reflection(data: np.ndarray, decision: np.ndarray) -> np.ndarray:
    """Conditionally reflects the data in XYZ.

    Args:
        data: input array, shape: [..], z, y, x, c
        decision: boolean array of shape 3, indicating reflections (x, y, z)

    Returns:
        Conditionally reflected array.
    """
    rank = data.ndim
    spatial_dims = [rank - 2, rank - 3, rank - 4]
    for i, do_flip in enumerate(decision):
        if do_flip:
            data = np.flip(data, axis=spatial_dims[i])
    return np.ascontiguousarray(data)


def xy_transpose(data: np.ndarray, decision: bool) -> np.ndarray:
    """Conditionally transposes the X and Y axes."""
    if not decision:
        return data
    rank = data.ndim
    perm = list(range(rank))
    perm[rank - 3], perm[rank - 2] = perm[rank - 2], perm[rank - 3]
    return np.transpose(data, perm)


def _elastic_warp_2d(patch, num_control_points_ratio,
                     deformation_stdev_ratio, mode='reflect'):
    """Applies 2D elastic deformation to all y,x slices of patch."""
    num_cp_y = max(int(num_control_points_ratio * patch.shape[1]), 1)
    num_cp_x = max(int(num_control_points_ratio * patch.shape[2]), 1)
    y = np.linspace(0, patch.shape[1], num_cp_y)
    x = np.linspace(0, patch.shape[2], num_cp_x)
    coords = np.array([(y0, x0) for y0 in y for x0 in x])
    deformation_stdev = deformation_stdev_ratio * np.min(patch.shape)
    deformations = np.random.normal(0, deformation_stdev, coords.shape)
    deformed_coords = coords + deformations
    grid_y, grid_x = np.mgrid[0:patch.shape[1], 0:patch.shape[2]]
    grid = griddata(coords, deformed_coords, (grid_y, grid_x),
                    method='cubic', fill_value=0)
    warped = np.zeros(patch.shape, dtype=patch.dtype)
    for b in range(patch.shape[0]):
        for c in range(patch.shape[3]):
            warped[b, :, :, c] = warp(
                patch[b, :, :, c],
                np.array((grid[:, :, 0], grid[:, :, 1])), mode=mode)
    return warped


def _affine_transform_2d(patch, rotation_max, scale_max, shear_max,
                         mode='reflect'):
    """Applies 2D affine transformation to all y,x slices of patch."""
    rotation = (np.random.rand() * 2 - 1) * rotation_max
    scale = 1 - (np.random.rand(2) * 2 - 1) * scale_max
    shear = (np.random.rand() * 2 - 1) * shear_max
    scale[1] *= np.cos(shear)
    at = AffineTransform(scale=scale, rotation=rotation, shear=shear)
    transformed = np.zeros(patch.shape, dtype=patch.dtype)
    for b in range(patch.shape[0]):
        for c in range(patch.shape[3]):
            transformed[b, :, :, c] = warp(
                patch[b, :, :, c], at, mode=mode)
    return transformed


def _apply_at_random_z_indices(patch, fn, max_indices_ratio):
    """Applies function to randomly selected z indices."""
    max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
    num_indices = np.random.randint(1, max_indices + 1)
    z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
    for z in z_indices:
        transformed = fn(patch[:, z, :, :, :].astype(np.float64))
        patch[:, z, :, :, :] = transformed.astype(patch.dtype)
    return patch, z_indices


def elastic_warp(patch, max_indices_ratio, num_control_points_ratio,
                 deformation_stdev_ratio, skip_ratio=0, mode='reflect'):
    """Applies elastic deformation to selected z indices."""
    patch = patch.copy()
    if np.random.rand() < skip_ratio:
        return [patch, -1]

    def warp_fn(p):
        return _elastic_warp_2d(p, num_control_points_ratio,
                                deformation_stdev_ratio, mode=mode)
    return _apply_at_random_z_indices(patch, warp_fn, max_indices_ratio)


def affine_transform(patch, max_indices_ratio, rotation_max, scale_max,
                     shear_max, skip_ratio=0, mode='reflect'):
    """Applies affine transform to selected z indices."""
    patch = patch.copy()
    if np.random.rand() < skip_ratio:
        return [patch, -1]

    def transform_fn(p):
        return _affine_transform_2d(p, rotation_max, scale_max,
                                    shear_max, mode=mode)
    return _apply_at_random_z_indices(patch, transform_fn, max_indices_ratio)


def _center_crop(patch, zyx_cropped_shape):
    """Crops center z,y,x dimensions of patch."""
    diff = np.array(patch.shape[1:-1]) - np.array(zyx_cropped_shape)
    assert np.all(diff >= 0)
    start = diff // 2
    end = patch.shape[1:-1] - np.ceil(diff / 2.0).astype(int)
    return patch[:, start[0]:end[0], start[1]:end[1], start[2]:end[2], :]


def misalignment(patch, labels, mask, patch_final_zyx, labels_final_zyx,
                 mask_final_zyx, max_offset, slip_ratio, skip_ratio=0):
    """Performs slip and translation misalignment augmentations."""
    patch, labels, mask = patch.copy(), labels.copy(), mask.copy()
    if np.random.rand() < skip_ratio:
        return (_center_crop(patch, patch_final_zyx),
                _center_crop(labels, labels_final_zyx),
                _center_crop(mask, mask_final_zyx), -1)

    zyx_max = np.array([patch.shape, labels.shape, mask.shape]).max(axis=0)[1:-1]

    def _edge_pad(p, target_shape):
        diff = np.array(target_shape) - np.array(p.shape[1:-1])
        assert np.all(diff >= 0)
        pad = [[d // 2, int(np.ceil(d / 2.0))] for d in diff]
        pad = [[0, 0]] + pad + [[0, 0]]
        return np.pad(p, pad, mode='edge')

    padded = [_edge_pad(patch, zyx_max), _edge_pad(labels, zyx_max),
              _edge_pad(mask, zyx_max)]

    offset_y, offset_x = np.random.randint(-max_offset, max_offset + 1, 2)
    z_start = np.random.randint(0, zyx_max[0])
    is_slip = np.random.rand() < slip_ratio

    results = []
    for d in padded:
        if is_slip:
            d[:, z_start, :, :, :] = np.roll(d[:, z_start, :, :, :], offset_y, 1)
            d[:, z_start, :, :, :] = np.roll(d[:, z_start, :, :, :], -offset_x, 2)
        else:
            d[:, z_start:, :, :, :] = np.roll(d[:, z_start:, :, :, :], offset_y, 2)
            d[:, z_start:, :, :, :] = np.roll(d[:, z_start:, :, :, :], -offset_x, 3)
        results.append(d)

    results[0] = _center_crop(results[0], patch_final_zyx)
    results[1] = _center_crop(results[1], labels_final_zyx)
    results[2] = _center_crop(results[2], mask_final_zyx)
    results.append(z_start)
    return results


def missing_section(patch, max_indices_ratio, skip_ratio=0, fill_value=None,
                    max_fill_val=256, full_prob=0.5, quadrant_prob=0.5):
    """Performs missing section augmentation."""
    patch = patch.copy()
    if np.random.rand() < skip_ratio:
        return [patch, -1]
    max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
    num_indices = np.random.randint(1, max_indices + 1)
    z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
    fill_val = fill_value if fill_value is not None else np.random.rand() * max_fill_val
    for z in z_indices:
        if np.random.rand() < full_prob:
            patch[:, z, :, :, :] = fill_val
        else:
            _quadrant_replace(patch, z,
                              np.full(patch[:, 0, :, :, :].shape, fill_val,
                                      patch.dtype),
                              quadrant_prob)
    return patch, z_indices


def _quadrant_replace(patch, z, replacement, quadrant_prob):
    """Replaces randomly selected x,y quadrants at specified z index."""
    apply_quadrants = np.random.rand(4) < quadrant_prob
    y = np.random.randint(0, patch.shape[2])
    x = np.random.randint(0, patch.shape[3])
    if apply_quadrants[0]:
        patch[:, z, 0:y, 0:x, :] = replacement[:, 0:y, 0:x, :]
    if apply_quadrants[1]:
        patch[:, z, y:, 0:x, :] = replacement[:, y:, 0:x, :]
    if apply_quadrants[2]:
        patch[:, z, 0:y, x:, :] = replacement[:, 0:y, x:, :]
    if apply_quadrants[3]:
        patch[:, z, y:, x:, :] = replacement[:, y:, x:, :]


def out_of_focus_section(patch, max_indices_ratio, max_filter_stdev,
                         skip_ratio=0, full_prob=0.5, quadrant_prob=0.5):
    """Applies out-of-focus-section augmentation."""
    patch = patch.copy()
    if np.random.rand() < skip_ratio:
        return [patch, -1]
    max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
    num_indices = np.random.randint(1, max_indices + 1)
    z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
    filter_stdev = np.random.rand() * max_filter_stdev
    for z in z_indices:
        blurred = gaussian_filter(patch[:, z, :, :, :], filter_stdev)
        if np.random.rand() < full_prob:
            patch[:, z, :, :, :] = blurred
        else:
            _quadrant_replace(patch, z, blurred, quadrant_prob)
    return patch, z_indices


def grayscale_perturb(patch, max_contrast_factor, max_brightness_factor,
                      skip_ratio=0, max_val=255, full_prob=0.5):
    """Applies brightness/contrast adjustment and gamma correction."""
    patch = patch.copy()
    if np.random.rand() < skip_ratio:
        return patch, 0

    def perturb_fn(p):
        cf = 1 + (np.random.rand() - 0.5) * max_contrast_factor
        bf = (np.random.rand() - 0.5) * max_brightness_factor
        power = 2.0 ** (np.random.rand() * 2 - 1)
        normalized = p.astype(np.float32) / max_val
        adjusted = normalized * cf + bf
        gamma = np.clip(adjusted, 0, 1) ** power
        return (gamma * max_val).astype(p.dtype)

    if np.random.rand() < full_prob:
        return [perturb_fn(patch), np.array(1)]
    else:
        for z in range(patch.shape[1]):
            patch[:, z, :, :, :] = perturb_fn(patch[:, z, :, :, :])
        return patch, 1


def apply_section_augmentations(
    patch, labels, mask,
    patch_final_zyx, labels_final_zyx, mask_final_zyx,
    elastic_warp_skip_ratio, affine_transform_skip_ratio,
    misalignment_skip_ratio, missing_section_skip_ratio,
    outoffocus_skip_ratio, grayscale_skip_ratio,
    max_warp_indices_ratio, num_control_points_ratio,
    deformation_stdev_ratio, max_affine_transform_indices_ratio,
    rotation_max, scale_max, shear_max,
    max_xy_offset, slip_vs_translate_ratio,
    max_missing_section_indices_ratio,
    max_outoffocus_indices_ratio, max_filter_stdev,
    max_contrast_factor, max_brightness_factor):
    """Performs ssEM training set augmentations (pure NumPy)."""

    patch = np.array(patch)
    labels = np.array(labels)
    mask = np.array(mask)

    patch, _ = elastic_warp(
        patch, max_warp_indices_ratio, num_control_points_ratio,
        deformation_stdev_ratio, elastic_warp_skip_ratio)

    patch, _ = affine_transform(
        patch, max_affine_transform_indices_ratio,
        rotation_max, scale_max, shear_max, affine_transform_skip_ratio)

    results = misalignment(
        patch, labels, mask, patch_final_zyx, labels_final_zyx,
        mask_final_zyx, max_xy_offset, slip_vs_translate_ratio,
        misalignment_skip_ratio)
    patch, labels, mask = results[0], results[1], results[2]

    patch, _ = missing_section(
        patch, max_missing_section_indices_ratio, missing_section_skip_ratio)

    patch, _ = out_of_focus_section(
        patch, max_outoffocus_indices_ratio, max_filter_stdev,
        outoffocus_skip_ratio)

    patch, _ = grayscale_perturb(
        patch, max_contrast_factor, max_brightness_factor,
        grayscale_skip_ratio)

    return patch, labels, mask


def soften_labels(bool_labels, softness=0.05):
    """Converts boolean labels to soft float labels.

    Args:
        bool_labels: boolean ndarray
        softness: value for False (1-softness for True)

    Returns:
        float32 ndarray with softened label values
    """
    return np.where(bool_labels, 1.0 - softness, softness).astype(np.float32)
