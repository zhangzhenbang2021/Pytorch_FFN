"""FFN seed policies—choosing segmentation start positions (PyTorch).

In FFN inference, the seed policy decides where to start segmenting new objects.
Good seed positions are inside the object (ideally center), away from boundaries.

Policy types:

1. PolicyPeaks: local peaks of Euclidean distance transform (EDT)
   - Compute distance from unsegmented region to boundary
   - Pick positions with maximum distance (object centers)
   - Most commonly used strategy

2. PolicyPeaks2d: 2D version of PolicyPeaks
   - For strongly anisotropic data

3. PolicyGrid3d: uniform sampling on a 3D grid
   - Simple but may miss small objects

4. PolicyInvertOrigins: reverse search from existing object origins

All policies inherit BaseSeedPolicy and provide a unified iterator interface.
Canvas selects the policy via the seed_policy argument of segment_all().

This module is framework-agnostic (pure NumPy/SciPy/EDT), no PyTorch dependency.
"""

import itertools
import threading
from typing import Any, Sequence
import weakref

from absl import logging
import edt
import numpy as np
from scipy import ndimage
import skimage
import skimage.feature
import skimage.morphology

from . import storage


class BaseSeedPolicy:
    """Base class for seed policies."""

    def __init__(self, canvas, **kwargs):
        logging.info('Deleting unused BaseSeedPolicy kwargs: %s', kwargs)
        del kwargs
        self.canvas = weakref.proxy(canvas)
        self.coords = None
        self.idx = 0

    def init_coords(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        if self.coords is None:
            self.init_coords()
            if self.coords is None:
                raise StopIteration()
            if self.coords.size:
                margin = np.array(self.canvas.margin)[np.newaxis, ...]
                self.coords = self.coords[np.all(
                    (self.coords - margin >= 0) &
                    (self.coords + margin < self.canvas.shape),
                    axis=1), :]

        while self.idx < self.coords.shape[0]:
            curr = self.coords[self.idx, :]
            self.idx += 1
            return tuple(curr)

        raise StopIteration()

    def next(self):
        return self.__next__()

    def get_state(self, previous=False):
        if previous:
            return self.coords, max(0, self.idx - 1)
        return self.coords, self.idx

    def set_state(self, state):
        self.coords, self.idx = state

    def get_exclusion_mask(self):
        mask = self.canvas.segmentation > 0
        if self.canvas.restrictor is not None:
            if self.canvas.restrictor.mask is not None:
                mask |= self.canvas.restrictor.mask
            if self.canvas.restrictor.seed_mask is not None:
                mask |= self.canvas.restrictor.seed_mask
        return mask


def _find_peaks(distances, **kwargs):
    rng = np.random.RandomState(seed=42)
    idxs = skimage.feature.peak_local_max(
        distances + rng.rand(*distances.shape) * 1e-4, **kwargs)
    return idxs


class PolicyPeaks(BaseSeedPolicy):
    """Attempts to find points away from edges in the image."""

    _sem = threading.Semaphore(4)

    def init_coords(self):
        logging.info('peaks: starting')
        edges = ndimage.generic_gradient_magnitude(
            self.canvas.image.astype(np.float32), ndimage.sobel)
        sigma = 49.0 / 6.0
        thresh_image = np.zeros(edges.shape, dtype=np.float32)
        ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
        filt_edges = edges > thresh_image
        del edges, thresh_image

        mask = self.get_exclusion_mask()

        if self.canvas.restrictor is not None:
            if self.canvas.restrictor.mask is not None:
                filt_edges[self.canvas.restrictor.mask] = 1
            if self.canvas.restrictor.seed_mask is not None:
                filt_edges[self.canvas.restrictor.seed_mask] = 1

        if np.all(filt_edges == 1):
            return

        with PolicyPeaks._sem:
            logging.info('peaks: filtering done')
            dt = edt.edt(
                1 - filt_edges,
                anisotropy=self.canvas.voxel_size_zyx).astype(np.float32)
            logging.info('peaks: edt done')
            dt[mask] = -1
            dt[~np.isfinite(dt)] = -1
            idxs = _find_peaks(dt, min_distance=3, threshold_abs=0,
                               threshold_rel=0)
            idxs = np.array(sorted((z, y, x) for z, y, x in idxs))
            logging.info('peaks: found %d local maxima', idxs.shape[0])
            self.coords = idxs


class PolicyPeaks2d(BaseSeedPolicy):
    """Attempts to find points away from edges at each 2d slice."""

    def __init__(self, canvas, min_distance=7, threshold_abs=2.5,
                 sort_cmp='ascending', **kwargs):
        super().__init__(canvas, **kwargs)
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.sort_reverse = sort_cmp.strip().lower().startswith('de')

    def init_coords(self):
        logging.info('2d peaks: starting')
        for z in range(self.canvas.image.shape[0]):
            image_2d = self.canvas.image[z, :, :].astype(np.float32)
            edges = ndimage.generic_gradient_magnitude(image_2d, ndimage.sobel)
            sigma = 49.0 / 6.0
            thresh_image = np.zeros(edges.shape, dtype=np.float32)
            ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
            filt_edges = edges > thresh_image
            del edges, thresh_image

            if (self.canvas.restrictor is not None and
                    self.canvas.restrictor.mask is not None):
                filt_edges[self.canvas.restrictor.mask[z, :, :]] = 1

            dt = edt.edt(1 - filt_edges).astype(np.float32)
            idxs = _find_peaks(dt, min_distance=self.min_distance,
                               threshold_abs=self.threshold_abs,
                               threshold_rel=0)
            zs = np.full((idxs.shape[0], 1), z, dtype=np.int64)
            idxs = np.concatenate((zs, idxs), axis=1)
            logging.info('2d peaks: found %d local maxima at z=%d',
                         idxs.shape[0], z)
            self.coords = (np.concatenate((self.coords, idxs))
                           if z != 0 else idxs)

        self.coords = np.array(sorted(
            [(z, y, x) for z, y, x in self.coords],
            reverse=self.sort_reverse))
        logging.info('2d peaks: found %d total', self.coords.shape[0])


class PolicyFillEmptySpace(BaseSeedPolicy):
    """Selects points in unsegmented parts of the image."""

    def init_coords(self):
        logging.info('fill_empty: starting')
        dt = edt.edt(self.canvas.segmentation == 0).astype(np.float32)
        idxs = _find_peaks(dt, min_distance=2, threshold_abs=0.5,
                           threshold_rel=0)
        logging.info('fill_empty: found %d local maxima', idxs.shape[0])
        self.coords = np.array(sorted((z, y, x) for z, y, x in idxs))


class PolicyGrid3d(BaseSeedPolicy):
    """Points distributed on a uniform 3d grid."""

    def __init__(self, canvas, step=16,
                 offsets=(0, 8, 4, 12, 2, 10, 14), **kwargs):
        super().__init__(canvas, **kwargs)
        self.step = step
        self.offsets = offsets

    def init_coords(self):
        coords = []
        for offset in self.offsets:
            for z in range(offset, self.canvas.image.shape[0], self.step):
                for y in range(offset, self.canvas.image.shape[1], self.step):
                    for x in range(offset, self.canvas.image.shape[2], self.step):
                        coords.append((z, y, x))
        self.coords = np.array(coords)


class PolicyGrid2d(BaseSeedPolicy):
    """Points distributed on a uniform 2d grid."""

    def __init__(self, canvas, step=16,
                 offsets=(0, 8, 4, 12, 2, 6, 10, 14), **kwargs):
        super().__init__(canvas, **kwargs)
        self.step = step
        self.offsets = offsets

    def init_coords(self):
        coords = []
        for offset in self.offsets:
            for z in range(self.canvas.image.shape[0]):
                for y in range(offset, self.canvas.image.shape[1], self.step):
                    for x in range(offset, self.canvas.image.shape[2], self.step):
                        coords.append((z, y, x))
        self.coords = np.array(coords)
