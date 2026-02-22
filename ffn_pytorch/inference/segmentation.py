"""Segmentation data utilities (PyTorch).

This module provides pure NumPy post-processing for segmentation:

- clear_dust(): remove objects below a size threshold (denoise)
- reduce_id_bits(): choose smallest integer type for segment IDs
- clean_up(): connected components + denoise
- split_segmentation_by_intersection(): intersection of two segmentations

Connected-components dependency:
- Prefer connectomics.segmentation.labels if available
- Otherwise use scipy.ndimage.label as fallback
- _LabelsShim provides a unified interface
"""

import numpy as np
from scipy import ndimage

try:
    from connectomics.segmentation import labels as _labels
    _has_connectomics = True
except ImportError:
    _has_connectomics = False


def _split_disconnected_components(seg, connectivity=1):
    """Fallback for connectomics.segmentation.labels.split_disconnected_components."""
    struct = ndimage.generate_binary_structure(seg.ndim, connectivity)
    output = np.zeros_like(seg)
    next_id = 1
    for obj_id in np.unique(seg):
        if obj_id == 0:
            continue
        mask = seg == obj_id
        labeled, n = ndimage.label(mask, structure=struct)
        for cc_id in range(1, n + 1):
            output[labeled == cc_id] = next_id
            next_id += 1
    return output


class _LabelsShim:
    """Provides split_disconnected_components, either from connectomics or fallback."""
    @staticmethod
    def split_disconnected_components(seg, connectivity=1):
        if _has_connectomics:
            return _labels.split_disconnected_components(seg, connectivity)
        return _split_disconnected_components(seg, connectivity)

    @staticmethod
    def make_contiguous(seg):
        if _has_connectomics:
            return _labels.make_contiguous(seg)
        unique_ids = np.unique(seg)
        mapping = {old: new for new, old in enumerate(unique_ids)}
        out = np.zeros_like(seg)
        for old, new in mapping.items():
            out[seg == old] = new
        return out, mapping


labels = _LabelsShim()


def clear_dust(data: np.ndarray, min_size: int = 10):
    """Removes small objects from a segmentation array."""
    ids, sizes = np.unique(data, return_counts=True)
    small = ids[sizes < min_size]
    small_mask = np.isin(data.flat, small).ravel().reshape(data.shape)
    data[small_mask] = 0
    return data


def reduce_id_bits(segmentation: np.ndarray):
    """Reduces the number of bits used for IDs."""
    max_id = segmentation.max()
    if max_id <= np.iinfo(np.uint8).max:
        return segmentation.astype(np.uint8)
    elif max_id <= np.iinfo(np.uint16).max:
        return segmentation.astype(np.uint16)
    elif max_id <= np.iinfo(np.uint32).max:
        return segmentation.astype(np.uint32)
    return segmentation


def clean_up(seg: np.ndarray, split_cc=True, connectivity=1,
             min_size=0, return_id_map=False):
    """Runs connected components and removes small objects."""
    cc_to_orig, _ = clean_up_and_count(
        seg, split_cc, connectivity, min_size,
        compute_id_map=return_id_map, compute_counts=False)
    if return_id_map:
        return cc_to_orig


def clean_up_and_count(seg: np.ndarray, split_cc=True, connectivity=1,
                       min_size=0, compute_id_map=True, compute_counts=True):
    """Runs connected components and removes small objects, returns metadata."""
    if compute_id_map:
        seg_orig = seg.copy()

    if split_cc:
        seg[...] = labels.split_disconnected_components(seg, connectivity)
    if min_size > 0:
        clear_dust(seg, min_size)

    cc_to_orig, cc_to_count = None, None

    if compute_id_map or compute_counts:
        unique_result_tuple = np.unique(
            seg.ravel(), return_index=compute_id_map,
            return_counts=compute_counts)
        cc_ids = unique_result_tuple[0]
    if compute_id_map:
        cc_idx = unique_result_tuple[1]
        orig_ids = seg_orig.ravel()[cc_idx]
        cc_to_orig = dict(zip(cc_ids, orig_ids))
    if compute_counts:
        cc_counts = unique_result_tuple[-1]
        cc_to_count = dict(zip(cc_ids, cc_counts))

    return cc_to_orig, cc_to_count


def split_segmentation_by_intersection(a: np.ndarray, b: np.ndarray,
                                       min_size: int):
    """Computes the intersection of two segmentations."""
    if a.shape != b.shape:
        raise ValueError
    a = a.ravel()
    output_array = a
    b = b.ravel()

    def remap_input(x):
        if x.dtype != np.uint64:
            raise TypeError
        max_uint32 = 2**32 - 1
        max_id = x.max()
        orig_values_map = None
        if max_id > max_uint32:
            orig_values_map, x = np.unique(x, return_inverse=True)
            if len(orig_values_map) > max_uint32:
                raise ValueError('More than 2**32-1 unique labels not supported')
            x = np.asarray(x, dtype=np.uint64)
            if orig_values_map[0] != 0:
                orig_values_map = np.concatenate(
                    [np.array([0], dtype=np.uint64), orig_values_map])
                x[...] += 1
        return x, max_id, orig_values_map

    remapped_a, max_id, a_reverse_map = remap_input(a)
    remapped_b, _, _ = remap_input(b)

    intersection_ids = np.bitwise_or(remapped_a, remapped_b << 32)
    unique_joint, remapped_joint, joint_counts = np.unique(
        intersection_ids, return_inverse=True, return_counts=True)

    unique_a = np.bitwise_and(unique_joint, 0xFFFFFFFF)
    unique_b = unique_joint >> 32

    max_overlap_ids = dict()
    for la, lb, cnt in zip(unique_a, unique_b, joint_counts):
        new_pair = (lb, cnt)
        existing = max_overlap_ids.setdefault(la, new_pair)
        if existing[1] < cnt:
            max_overlap_ids[la] = new_pair

    new_labels = np.zeros(len(unique_joint), np.uint64)
    for i, (la, lb, cnt) in enumerate(
            zip(unique_a, unique_b, joint_counts)):
        if cnt < min_size or la == 0:
            new_label = 0
        elif lb == max_overlap_ids[la][0]:
            if a_reverse_map is not None:
                new_label = a_reverse_map[la]
            else:
                new_label = la
        else:
            max_id += 1
            new_label = max_id
        new_labels[i] = new_label

    output_array[...] = new_labels[remapped_joint]
