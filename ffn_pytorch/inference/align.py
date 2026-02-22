"""Classes to support ad-hoc alignment for inference (PyTorch version).

This module is already framework-independent (pure NumPy).
"""

import numpy as np


class Alignment:
    """Base class for local ad-hoc alignment (identity/no-op)."""

    def __init__(self, corner, size):
        self._corner = corner
        self._size = size

    @property
    def corner(self):
        return self._corner

    @property
    def size(self):
        return self._size

    def expand_bounds(self, corner, size, forward=True):
        del forward
        return corner, size

    def transform_shift_mask(self, corner, scale, mask):
        del corner, scale
        return mask

    def align_and_crop(self, src_corner, source, dst_corner, dst_size,
                       fill=0, forward=True):
        del forward

        if (np.all(np.array(src_corner) == np.array(dst_corner)) and
                np.all(np.array(source.shape) == np.array(dst_size))):
            return source

        destination = np.full(dst_size, fill, dtype=source.dtype)
        zyx_offset = np.array(src_corner) - np.array(dst_corner)
        src_size = np.array(source.shape)

        dst_beg = np.clip(zyx_offset, 0, dst_size).astype(int)
        dst_end = np.clip(dst_size, 0, src_size + zyx_offset).astype(int)
        src_beg = np.clip(-zyx_offset, 0, src_size).astype(int)
        src_end = np.clip(src_size, 0, dst_size - zyx_offset).astype(int)

        if np.any(dst_end - dst_beg == 0) or np.any(src_end - src_beg == 0):
            return destination

        destination[dst_beg[0]:dst_end[0],
                    dst_beg[1]:dst_end[1],
                    dst_beg[2]:dst_end[2]] = source[src_beg[0]:src_end[0],
                                                    src_beg[1]:src_end[1],
                                                    src_beg[2]:src_end[2]]
        return destination

    def transform(self, zyx, forward=True):
        del forward
        return zyx

    def rescaled(self, zyx_scale):
        zyx_scale = np.array(zyx_scale)
        return Alignment(zyx_scale * self.corner, zyx_scale * self.size)


class Aligner:
    """Base class for alignment generators (identity)."""

    def generate_alignment(self, corner, size):
        return Alignment(corner, size)
