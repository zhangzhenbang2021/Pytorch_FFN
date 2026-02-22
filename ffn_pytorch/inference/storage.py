"""FFN inference storage and I/O (PyTorch).

This module provides read/write for inference data:

1. Volume loading (decorated_volume):
   - HDF5: open via h5py, path format "file.h5:dataset"
   - TensorStore: open via tensorstore, JSON config
   - SyncAdapter wraps TensorStore for synchronous NumPy-style indexing

2. Segmentation I/O:
   - save_subvolume(): save as NPZ (segmentation, origins, request config)
   - load_segmentation(): load with optional post-processing (threshold, connected components)
   - Probability map quantize/dequantize (float32 ↔ uint8)

3. Atomic file write (atomic_file):
   - Write to temp file, then atomically move to target path
   - Avoids half-written files on interrupt

4. Path helpers:
   - subvolume_path(): hierarchical dirs {z}/{y}/seg-{x}_{y}_{z}.npz
   - checkpoint_path/segmentation_path/object_prob_path: per-type paths

5. Mask building (build_mask):
   - Coordinate expressions (coordinate_expression)
   - Volume channels, image channels
   - Value range filter and inversion

Original TF used tensorflow.io.gfile; PyTorch version uses standard Python os/shutil.

NumpyArray: ndarray subclass with clear() method, used for Canvas seed/segmentation/seg_prob.
"""

from collections import namedtuple
from contextlib import contextmanager
import logging
import json
import os
import re
import shutil
import tempfile
from typing import Any, Optional

import h5py
import numpy as np

from . import align
from . import segmentation

OriginInfo = namedtuple('OriginInfo', ['start_zyx', 'iters', 'walltime_sec'])
Volume = Any


class SyncAdapter:
    """Makes it possible to use a TensorStore as a numpy array synchronously."""

    def __init__(self, tstore):
        self.tstore = tstore

    def __getitem__(self, ind):
        return np.array(self.tstore[ind])

    def __getattr__(self, attr):
        return getattr(self.tstore, attr)

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.tstore)})'


class NumpyArray(np.ndarray):
    """ndarray with a clear method."""

    def __new__(cls, default_value=0, **kwargs):
        ret = super(NumpyArray, cls).__new__(cls, **kwargs)
        ret.default_value = default_value
        return ret

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.clear()

    def clear(self):
        self[...] = self.default_value


def decorated_volume(settings, **kwargs) -> Volume:
    """Converts DecoratedVolume proto object into volume objects."""
    del kwargs

    if settings.HasField('volinfo'):
        raise NotImplementedError('VolumeStore operations not available.')
    elif settings.HasField('hdf5'):
        path = settings.hdf5.split(':')
        if len(path) != 2:
            raise ValueError(
                'hdf5 volume_path should be file_path:dataset_path. Got: '
                + settings.hdf5)
        volume = h5py.File(path[0], 'r')[path[1]]
    elif settings.HasField('tensorstore'):
        try:
            import tensorstore as ts
            volume = SyncAdapter(ts.open(json.loads(settings.tensorstore)).result())
        except ImportError:
            raise ImportError('tensorstore package required for tensorstore volumes.')
    else:
        raise ValueError('A volume_path must be set.')

    if volume.ndim not in (3, 4):
        raise ValueError('Volume must be 3d or 4d.')

    return volume


@contextmanager
def atomic_file(path, mode='w+b'):
    """Atomically saves data to a target path."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with tempfile.NamedTemporaryFile(mode=mode, delete=False) as tmp:
        yield tmp
        tmp.flush()
    tmp_path = tmp.name + '.tmp'
    shutil.move(tmp.name, tmp_path)
    shutil.move(tmp_path, path)


def quantize_probability(prob: np.ndarray) -> np.ndarray:
    """Quantizes a probability map into a byte array."""
    ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))
    ret[np.isnan(prob)] = 0
    return ret.astype(np.uint8)


def dequantize_probability(prob: np.ndarray) -> np.ndarray:
    """Dequantizes a byte array representing a probability map."""
    dq = 1.0 / 255
    ret = ((prob - 0.5) * dq).astype(np.float32)
    ret[prob == 0] = np.nan
    return ret


def save_subvolume(labels, origins, output_path, **misc_items):
    """Saves an FFN subvolume."""
    seg = segmentation.reduce_id_bits(labels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with atomic_file(output_path) as fd:
        np.savez_compressed(fd, segmentation=seg, origins=origins,
                            **misc_items)


def subvolume_path(output_dir, corner, suffix):
    """Returns path to a file with FFN subvolume data."""
    return os.path.join(
        output_dir, str(corner[2]), str(corner[1]),
        'seg-%s.%s' % ('_'.join([str(x) for x in corner[::-1]]), suffix))


def legacy_subvolume_path(output_dir, corner, suffix):
    return os.path.join(output_dir, 'seg-%s.%s' % (
        '_'.join([str(x) for x in corner[::-1]]), suffix))


def get_corner_from_path(path):
    """Returns subvolume corner as (z, y, x)."""
    match = re.search(r'(\d+)_(\d+)_(\d+).npz', os.path.basename(path))
    if match is None:
        raise ValueError('Unrecognized path: %s' % path)
    coord = tuple([int(x) for x in match.groups()])
    return coord[::-1]


def get_existing_corners(segmentation_dir):
    import glob as glob_mod
    corners = []
    for path in glob_mod.glob(os.path.join(segmentation_dir, 'seg-*_*_*.npz')):
        corners.append(get_corner_from_path(path))
    for path in glob_mod.glob(os.path.join(segmentation_dir, '*/*/seg-*_*_*.npz')):
        corners.append(get_corner_from_path(path))
    return corners


def checkpoint_path(output_dir, corner):
    return subvolume_path(output_dir, corner, 'cpoint')


def segmentation_path(output_dir, corner):
    return subvolume_path(output_dir, corner, 'npz')


def object_prob_path(output_dir, corner):
    return subvolume_path(output_dir, corner, 'prob')


def legacy_segmentation_path(output_dir, corner):
    return legacy_subvolume_path(output_dir, corner, 'npz')


def legacy_object_prob_path(output_dir, corner):
    return legacy_subvolume_path(output_dir, corner, 'prob')


def get_existing_subvolume_path(segmentation_dir, corner,
                                allow_cpoint=False):
    target_path = segmentation_path(segmentation_dir, corner)
    if os.path.exists(target_path):
        return target_path

    target_path = legacy_segmentation_path(segmentation_dir, corner)
    if os.path.exists(target_path):
        return target_path

    if allow_cpoint:
        target_path = checkpoint_path(segmentation_dir, corner)
        if os.path.exists(target_path):
            return target_path

    return None


def clip_subvolume_to_bounds(corner, size, volume):
    """Clips a subvolume bounding box to the image volume store bounds."""
    volume_size = np.array(volume.shape)
    if volume.ndim == 4:
        volume_size = volume_size[1:]
    corner = np.array(corner)
    size = np.array(size)
    clipped_start = np.maximum(corner, 0)
    clipped_end = np.minimum(corner + size, volume_size)
    clipped_size = np.maximum(clipped_end - clipped_start, 0)
    return clipped_start, clipped_size


def build_mask(masks, corner, subvol_size, mask_volume_map=None,
               image=None, alignment=None):
    """Builds a boolean mask."""
    final_mask = None
    if mask_volume_map is None:
        mask_volume_map = {}

    if alignment is None:
        alignment = align.Alignment(corner, subvol_size)

    src_corner, src_size = alignment.expand_bounds(
        corner, subvol_size, forward=False)

    for config in masks:
        curr_mask = np.zeros(subvol_size, dtype=bool)

        source_type = config.WhichOneof('source')
        if source_type == 'coordinate_expression':
            z, y, x = np.mgrid[
                src_corner[0]:src_corner[0] + src_size[0],
                src_corner[1]:src_corner[1] + src_size[1],
                src_corner[2]:src_corner[2] + src_size[2]]
            bool_mask = eval(config.coordinate_expression.expression)
            curr_mask |= alignment.align_and_crop(
                src_corner, bool_mask, corner, subvol_size)
        else:
            if source_type == 'image':
                assert image is not None
                channels = config.image.channels
                mask = image[np.newaxis, ...]
            elif source_type == 'volume':
                channels = config.volume.channels
                volume_key = config.volume.mask.SerializeToString()
                if volume_key not in mask_volume_map:
                    mask_volume_map[volume_key] = decorated_volume(
                        config.volume.mask)
                volume = mask_volume_map[volume_key]
                clipped_corner, clipped_size = clip_subvolume_to_bounds(
                    src_corner, src_size, volume)
                clipped_end = clipped_corner + clipped_size
                mask = volume[:, clipped_corner[0]:clipped_end[0],
                              clipped_corner[1]:clipped_end[1],
                              clipped_corner[2]:clipped_end[2]]
            else:
                logging.fatal('Unsupported mask source: %s', source_type)

            for chan_config in channels:
                channel_mask = mask[chan_config.channel, ...]
                channel_mask = alignment.align_and_crop(
                    src_corner, channel_mask, corner, subvol_size)
                if chan_config.values:
                    bool_mask = np.isin(
                        channel_mask, chan_config.values).ravel().reshape(
                            channel_mask.shape)
                else:
                    bool_mask = ((channel_mask >= chan_config.min_value) &
                                 (channel_mask <= chan_config.max_value))
                if chan_config.invert:
                    bool_mask = np.logical_not(bool_mask)
                curr_mask |= bool_mask

        if config.invert:
            curr_mask = np.logical_not(curr_mask)

        if final_mask is None:
            final_mask = curr_mask
        else:
            final_mask |= curr_mask

    return final_mask


def load_segmentation(segmentation_dir, corner, allow_cpoint=False,
                      threshold=None, split_cc=True, min_size=0,
                      mask_config=None):
    """Loads segmentation from an FFN subvolume."""
    target_path = get_existing_subvolume_path(
        segmentation_dir, corner, allow_cpoint)
    if target_path is None:
        raise ValueError('Segmentation not found, %s, %r.' %
                         (segmentation_dir, corner))

    with open(target_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        if 'segmentation' in data:
            seg = data['segmentation']
        else:
            raise ValueError('Invalid FFN NPZ file: %s' % target_path)

        origins = data['origins'].item()
        output = seg.astype(np.uint64)

        logging.info('loading segmentation from: %s', target_path)

        if threshold is not None:
            logging.info('thresholding at %f', threshold)
            prob_path = object_prob_path(segmentation_dir, corner)
            if not os.path.exists(prob_path):
                prob_path = legacy_object_prob_path(segmentation_dir, corner)
            with open(prob_path, 'rb') as pf:
                pdata = np.load(pf)
                prob = dequantize_probability(pdata['qprob'])
                output[prob < threshold] = 0

        if mask_config is not None:
            mask = build_mask(mask_config.masks, corner, seg.shape)
            output[mask] = 0

        if split_cc or min_size:
            new_to_old = segmentation.clean_up(
                output, split_cc, min_size, return_id_map=True)
            new_origins = {}
            for new_id, old_id in new_to_old.items():
                if old_id in origins:
                    new_origins[new_id] = origins[old_id]
            origins = new_origins

    return output, origins
