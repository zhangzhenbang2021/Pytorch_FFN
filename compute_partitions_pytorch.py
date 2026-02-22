#!/usr/bin/env python
# Copyright 2017-2023 Google Inc.
# ==============================================================================

r"""Computes the partition map for a segmentation (Optimized PyTorch GPU Version).
"""
from absl import app
from absl import flags
from absl import logging

from ffn.inference import segmentation
from ffn.inference import storage
from ffn.utils import bounding_box

import h5py
import numpy as np
import torch
import torch.nn.functional as F

FLAGS = flags.FLAGS

flags.DEFINE_string('input_volume', None, 'Segmentation volume...')
flags.DEFINE_string('output_volume', None, 'Output volume...')
flags.DEFINE_list('thresholds', None, 'List of thresholds...')
flags.DEFINE_list('lom_radius', None, 'LOM radii as (x, y, z)...')
flags.DEFINE_list('id_whitelist', None, 'Whitelist of object IDs...')
flags.DEFINE_list('exclusion_regions', None, 'Exclusion regions...')
flags.DEFINE_string('mask_configs', None, 'MaskConfigs proto...')
flags.DEFINE_integer('min_size', 10000, 'Minimum size...')


def load_mask(mask_configs, box, lom_diam_zyx):
    if mask_configs is None:
        return None

    mask = storage.build_mask(mask_configs.masks, box.start[::-1],
                              box.size[::-1])
    
    # Local CPU integral image (MaskConfigs only)
    val = np.ascontiguousarray(mask).astype(np.int32)
    svt = np.cumsum(val, axis=0)
    np.cumsum(svt, axis=1, out=svt)
    np.cumsum(svt, axis=2, out=svt)
    out = np.zeros((svt.shape[0] + 1, svt.shape[1] + 1, svt.shape[2] + 1), dtype=np.int32)
    out[1:, 1:, 1:] = svt
    
    svt = out
    diam = lom_diam_zyx
    summed = (svt[diam[0]:, diam[1]:, diam[2]:] - svt[diam[0]:, diam[1]:, :-diam[2]] -
              svt[diam[0]:, :-diam[1], diam[2]:] - svt[:-diam[0], diam[1]:, diam[2]:] +
              svt[:-diam[0], :-diam[1], diam[2]:] + svt[:-diam[0], diam[1]:, :-diam[2]] +
              svt[diam[0]:, :-diam[1], :-diam[2]] - svt[:-diam[0], :-diam[1], :-diam[2]])
              
    return summed >= 1


def clear_dust_gpu(data_t, min_size=10):
    ids, counts = torch.unique(data_t, return_counts=True)
    small_ids = ids[counts < min_size]
    
    if len(small_ids) > 0:
        small_mask = torch.isin(data_t, small_ids)
        data_t.masked_fill_(small_mask, 0)
        del small_mask  # Free after use
        
    del ids, counts, small_ids
    return data_t


def compute_partitions_gpu(seg_array, thresholds, lom_radius,
                           id_whitelist=None, exclusion_regions=None,
                           mask_configs=None, min_size=10000):
                           
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seg_tensor = torch.tensor(seg_array, device=device)
    seg_tensor = clear_dust_gpu(seg_tensor, min_size=min_size)
    assert seg_tensor.ndim == 3

    lom_radius = np.array(lom_radius)
    lom_radius_zyx = lom_radius[::-1]
    lom_diam_zyx = 2 * lom_radius_zyx + 1

    def _sel(i):
        return slice(None) if i == 0 else slice(i, -i)

    valid_sel = tuple(_sel(x) for x in lom_radius_zyx)
    output_shape = seg_tensor[valid_sel].shape
    output = torch.zeros(output_shape, dtype=torch.uint8, device=device)
    corner = lom_radius

    # Handle exclusion regions
    if exclusion_regions is not None:
        sz, sy, sx = output_shape
        hz = torch.arange(sz, device=device) + corner[2]
        hy = torch.arange(sy, device=device) + corner[1]
        hx = torch.arange(sx, device=device) + corner[0]
        hz, hy, hx = torch.meshgrid(hz, hy, hx, indexing='ij')

        for x, y, z, r in exclusion_regions:
            mask = (hx - x)**2 + (hy - y)**2 + (hz - z)**2 <= r**2
            output.masked_fill_(mask, 255)
            
        del hz, hy, hx, mask

    counts = torch.bincount(seg_tensor.view(-1))
    labels = torch.nonzero(counts).squeeze(-1).cpu().numpy()
    del counts
    
    if id_whitelist is not None:
        whitelist_set = set(id_whitelist)
        labels = [l for l in labels if l in whitelist_set]

    mask_np = load_mask(mask_configs,
                        bounding_box.BoundingBox(start=(0, 0, 0), size=seg_array.shape[::-1]),
                        lom_diam_zyx)
    if mask_np is not None:
        mask_tensor = torch.tensor(mask_np, device=device)
        output.masked_fill_(mask_tensor, 255)
        del mask_tensor

    fov_volume = float(np.prod(lom_diam_zyx))
    thr_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)
    dz, dy, dx = lom_diam_zyx

    for l in labels:
        if l == 0:
            continue

        # 1. Build object mask (int32 for integral image)
        object_mask = (seg_tensor == l).to(torch.int32)

        # 2. 3D integral image (SVT)
        svt = object_mask.cumsum(dim=0).cumsum(dim=1).cumsum(dim=2)
        svt = F.pad(svt, (1, 0, 1, 0, 1, 0), mode='constant', value=0)

        # 3. Fast box-sum query
        active_sum = (svt[dz:, dy:, dx:]
                    - svt[:-dz, dy:, dx:]
                    - svt[dz:, :-dy, dx:]
                    + svt[:-dz, :-dy, dx:]
                    - svt[dz:, dy:, :-dx]
                    + svt[:-dz, dy:, :-dx]
                    + svt[dz:, :-dy, :-dx]
                    - svt[:-dz, :-dy, :-dx])
        
        # Free large svt tensor
        del svt 

        active_fraction = active_sum.to(torch.float32) / fov_volume
        del active_sum 
        
        # 4. Build valid slice mask
        object_mask_valid = object_mask[valid_sel].to(torch.bool)
        del object_mask 

        write_mask = object_mask_valid & (output == 0)
        del object_mask_valid 

        # 5. Bucketize partition assignment
        binmap = torch.bucketize(active_fraction, thr_tensor, right=True).to(output.dtype) + 1
        del active_fraction 

        # 6. In-place mask write
        output[write_mask] = binmap[write_mask]
        
        del write_mask, binmap

    logging.info('finished processing %d labels', len(labels))

    # Clean up tensors
    del seg_tensor, thr_tensor
    torch.cuda.empty_cache()

    return corner, output.cpu().numpy()


def adjust_bboxes(bboxes, lom_radius):
    ret = []
    for bbox in bboxes:
        bbox = bbox.adjusted_by(start=lom_radius, end=-lom_radius)
        if np.all(bbox.size > 0):
            ret.append(bbox)
    return ret


def main(argv):
    del argv  
    
    path, dataset = FLAGS.input_volume.split(':')
    with h5py.File(path) as f:
        segmentation = f[dataset]
        bboxes = []
        for name, v in segmentation.attrs.items():
            if name.startswith('bounding_boxes'):
                for bbox in v:
                    bboxes.append(bounding_box.BoundingBox(bbox[0], bbox[1]))

        if not bboxes:
            bboxes.append(bounding_box.BoundingBox(start=(0, 0, 0), size=segmentation.shape[::-1]))

        shape = segmentation.shape
        lom_radius = [int(x) for x in FLAGS.lom_radius]
        
        corner, partitions = compute_partitions_gpu(
            segmentation[...], [float(x) for x in FLAGS.thresholds], lom_radius,
            FLAGS.id_whitelist, FLAGS.exclusion_regions, FLAGS.mask_configs,
            FLAGS.min_size)

    bboxes = adjust_bboxes(bboxes, np.array(lom_radius))

    path, dataset = FLAGS.output_volume.split(':')
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset(dataset, shape=shape, dtype=np.uint8, fillvalue=255,
                              chunks=True, compression='gzip')
        s = partitions.shape
        ds[corner[2]:corner[2] + s[0],
           corner[1]:corner[1] + s[1],
           corner[0]:corner[0] + s[2]] = partitions
           
        ds.attrs['bounding_boxes'] = [(b.start, b.size) for b in bboxes]
        ds.attrs['partition_counts'] = np.array(np.unique(partitions, return_counts=True))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_volume')
    flags.mark_flag_as_required('output_volume')
    flags.mark_flag_as_required('thresholds')
    flags.mark_flag_as_required('lom_radius')
    app.run(main)