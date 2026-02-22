"""FFN mask crop, pad, and seed creation (PyTorch).

This module provides low-level array ops for FFN training:

- crop(): crop subvolume at center offset from a larger array
- crop_and_pad(): crop with optional zero-pad to target size
- update_at(): write new values at given offset in array
- make_seed(): create center single-voxel seed array

Used heavily in training sample generation: on each FOV move,
crop FOV-sized subvolumes from full labels/patches/seed.

Coordinate convention:
- offset in (x, y, z) or (x, y) order
- array shape in (batch, z, y, x, channel) order
- crop_shape in (z, y, x) order (matches array axes)

Original TF crop() used tf.slice; PyTorch version uses pure NumPy slicing.
"""

from typing import Optional, Sequence

import numpy as np


def crop(data: np.ndarray, offset: Sequence[int],
         crop_shape: Sequence[int]) -> np.ndarray:
    """Extracts 'crop_shape' around 'offset' from 'data'.

    Pure NumPy replacement for the TF version.

    Args:
        data: array to extract from (b, [z], y, x, c)
        offset: (x, y, [z]) offset from center of 'data'
        crop_shape: (x, y, [z]) shape to extract

    Returns:
        cropped array
    """
    shape = list(data.shape)

    if shape[1:-1] == list(crop_shape[::-1]):
        return data

    off_y = shape[-3] // 2 - crop_shape[1] // 2 + offset[1]
    off_x = shape[-2] // 2 - crop_shape[0] // 2 + offset[0]

    if len(offset) == 2:
        return data[:, off_y:off_y + crop_shape[1],
                     off_x:off_x + crop_shape[0], :]
    else:
        off_z = shape[-4] // 2 - crop_shape[2] // 2 + offset[2]
        return data[:, off_z:off_z + crop_shape[2],
                     off_y:off_y + crop_shape[1],
                     off_x:off_x + crop_shape[0], :]


def update_at(to_update, offset, new_value, valid=None):
    """Pastes 'new_value' into 'to_update'.

    Args:
        to_update: numpy array to update (b, [z], y, x, c)
        offset: (x, y, [z]) offset from the center point of 'to_update'
        new_value: numpy array with values to paste (b, [z], y, x, c)
        valid: optional mask over batch dimension
    """
    shape = np.array(to_update.shape[1:-1])
    crop_shape_arr = np.array(new_value.shape[1:-1])
    offset_arr = np.array(offset[::-1])

    start = shape // 2 - crop_shape_arr // 2 + offset_arr
    end = start + crop_shape_arr

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector + [slice(None)])

    if valid is not None:
        to_update[selector][valid] = new_value[valid]
    else:
        to_update[selector] = new_value


def crop_and_pad(data: np.ndarray,
                 offset: Sequence[int],
                 crop_shape: Sequence[int],
                 target_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Extracts 'crop_shape' around 'offset' from 'data'.

    Optionally pads with zeros to 'target_shape'.

    Args:
        data: D+2 or higher-dim array, shape (b, [z], y, x, c)
        offset: (x, y, [z]) offset from center of 'data'
        crop_shape: ([cz], cy, cx) shape to extract
        target_shape: optional ([tz], ty, tx) shape to pad to

    Returns:
        Extracted (and optionally padded) array
    """
    dim = len(offset)

    shape = np.array(data.shape[-(1 + dim):-1])
    crop_shape = np.array(crop_shape)
    offset_arr = np.array(offset[::-1])

    start = shape // 2 - crop_shape // 2 + offset_arr
    end = start + crop_shape

    num_batch = len(data.shape) - dim - 1

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] * num_batch + selector + [slice(None)])
    cropped = data[selector]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - crop_shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = [(0, 0)] * num_batch
        paddings.extend(zip(pre, post))
        paddings.append((0, 0))

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def make_seed(shape, batch_size, pad=0.05, seed=0.95):
    """Builds a numpy array with a single voxel seed in the center.

    Args:
        shape: spatial size of the seed array (z, y, x)
        batch_size: batch dimension size
        pad: value used where the seed is inactive
        seed: value used where the seed is active

    Returns:
        float32 ndarray of shape [b, z, y, x, 1] with the seed
    """
    seed_array = np.full([batch_size] + list(shape) + [1], pad,
                         dtype=np.float32)
    idx = tuple([slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed
    return seed_array
