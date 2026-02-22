"""FFN training data loading (PyTorch).

This module loads training data from disk:
1. Training coordinates: where to extract training samples
2. Image/label volumes: extract 3D subvolumes from HDF5 datasets
3. Preprocessing: label softening, image normalization, augmentation

Differences from original TF version:
- TF uses TFRecord + tf.data.Dataset + tf.placeholder pipeline
- PyTorch uses HDF5 + Python generators + manual NumPy conversion
- TFRecord reading kept for backward compatibility (requires TF)

Data flow:
    Coordinate file (HDF5/TFRecord)
        → CoordinateLoader (random sampling)
        → load_from_numpylike (extract subvolumes from HDF5)
        → soften_labels + augmentation (preprocessing)
        → offset_and_scale_patches (normalization)
"""

import glob
import os
import random
from typing import Optional, Sequence

import h5py
import numpy as np

from . import augmentation


import h5py
import numpy as np

def load_coordinates_from_h5(coord_path: str) -> list:
    """Loads training coordinates from an HDF5 file (compatible with indexed versions).

    Args:
        coord_path: Path to the HDF5 file containing 'coords', 'vol_indices', 
                   and the 'label_map' attribute.

    Returns:
        A list of (center_xyz, volname) tuples.
    """
    with h5py.File(coord_path, 'r') as f:
        # 1. Load coordinates array, expected shape: (N, 3)
        centers = f['coords'][:]
        
        # 2. Handle volume labeling logic
        if 'vol_indices' in f and 'label_map' in f.attrs:
            # Case A: Using 'vol_indices' [0, 1, 0...] and an attribute 'label_map'
            print("Found 'vol_indices' and 'label_map' in HDF5. Mapping indices to volume names.")
            
            vol_indices = f['vol_indices'][:]
            
            # Decode byte strings from attributes into a UTF-8 list
            # e.g., [b'vol1', b'vol2'] -> ['vol1', 'vol2']
            label_map = [lbl.decode('utf-8') for lbl in f.attrs['label_map']]
            print(f"Loaded label map with {len(label_map)} entries from HDF5 attributes.")
            
            # Efficiently map indices back to string names
            volnames = [label_map[i] for i in vol_indices]
            print(f"Successfully mapped {len(vol_indices)} volume indices.")
            
        elif 'label_volume_names' in f:
            # Case B: Direct list of volume names stored in the dataset
            volnames = [v.decode('utf-8') if isinstance(v, bytes) else v
                        for v in f['label_volume_names'][:]]
        else:
            # Fallback if no labeling info is found
            raise KeyError("Could not find volume labeling information (vol_indices or label_volume_names) in HDF5.")

    # Combine into a list of tuples: [(array([x,y,z]), "name"), ...]
    return list(zip(centers, volnames))


def load_coordinates_from_tfrecord(coord_pattern: str) -> list:
    """Loads training coordinates from TFRecord files.

    Uses TF only for reading the existing TFRecord format.
    Falls back if TF is not available.

    Args:
        coord_pattern: glob pattern for TFRecord files

    Returns:
        list of (center_xyz, volname) tuples
    """
    try:
        import tensorflow.compat.v1 as tf
        tf.enable_eager_execution()
    except Exception:
        raise RuntimeError(
            'TFRecord loading requires TensorFlow. '
            'Convert to HDF5 using build_coordinates_pytorch.py first.')

    coords = []
    files = sorted(glob.glob(coord_pattern))
    record_options = tf.io.TFRecordOptions(
        compression_type='GZIP')

    for filepath in files:
        dataset = tf.data.TFRecordDataset(filepath, compression_type='GZIP')
        for raw_record in dataset:
            example = tf.io.parse_single_example(raw_record, features={
                'center': tf.io.FixedLenFeature([1, 3], tf.int64),
                'label_volume_name': tf.io.FixedLenFeature([1], tf.string),
            })
            center = example['center'].numpy().squeeze()
            volname = example['label_volume_name'].numpy()[0].decode('utf-8')
            coords.append((center, volname))

    return coords


class CoordinateLoader:
    """Loads training coordinates from files and yields them randomly."""

    def __init__(self, coord_pattern: str, shuffle: bool = True):
        if coord_pattern.endswith('.h5') or coord_pattern.endswith('.hdf5'):
            files = sorted(glob.glob(coord_pattern))
            self.coords = []
            for f in files:
                self.coords.extend(load_coordinates_from_h5(f))
        else:
            self.coords = load_coordinates_from_tfrecord(coord_pattern)

        if shuffle:
            random.shuffle(self.coords)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.coords):
            random.shuffle(self.coords)
            self._idx = 0
        center, volname = self.coords[self._idx]
        self._idx += 1
        return center, volname


def load_from_numpylike(coord_xyz, volume_name, shape_xyz, volume_map):
    """Loads data from numpy-like volumes.

    Args:
        coord_xyz: [3] array with XYZ coordinates of the subvolume center
        volume_name: string name of the volume
        shape_xyz: [3] XYZ shape of the data to load
        volume_map: dict mapping volume names to volume objects

    Returns:
        ndarray of shape [1, z, y, x, c]
    """
    volume = volume_map[volume_name]
    start_offset = (np.array(shape_xyz) - 1) // 2
    starts = np.array(coord_xyz) - start_offset

    # ZYX slice from xyz start/size
    ends = starts + np.array(shape_xyz)
    slc = np.index_exp[starts[2]:ends[2], starts[1]:ends[1], starts[0]:ends[0]]

    if volume.ndim == 4:
        slc = np.index_exp[:] + slc
    data = volume[slc]

    if data.ndim == 4:
        data = np.rollaxis(data, 0, start=4)
    else:
        data = np.expand_dims(data, data.ndim)

    data = np.expand_dims(data, 0)
    return data


def soften_labels(bool_labels, softness=0.05):
    """Converts boolean labels into float32."""
    return np.where(bool_labels, 1.0 - softness, softness).astype(np.float32)


def offset_and_scale_patches(patches, volname, offset_scale_map=None,
                             default_offset=0.0, default_scale=1.0):
    """Apply offset and scale normalization to patches.

    Args:
        patches: ndarray of image data
        volname: volume name string
        offset_scale_map: optional dict mapping volname to (offset, scale)
        default_offset: default image mean
        default_scale: default image stddev

    Returns:
        normalized float32 patches
    """
    if offset_scale_map and volname in offset_scale_map:
        offset, scale = offset_scale_map[volname]
    else:
        offset = default_offset
        scale = default_scale

    return (patches.astype(np.float32) - offset) / scale


def load_example(coord_loader, label_volume_map, image_volume_map,
                 label_size_xyz, image_size_xyz,
                 image_mean, image_stddev,
                 offset_scale_map=None,
                 permutable_axes=None, reflectable_axes=None):
    """Loads a single training example with FFN-specific label processing.

    Returns:
        tuple: (patch, labels, loss_weights, coord, volname)
    """
    # coord_xyz: [3] (x, y, z)
    coord_xyz, volname = next(coord_loader)

    # Load 3D patches from HDF5
    # labels shape: [1, Z, Y, X, 1] (Batch=1, Channels=1)
    labels = load_from_numpylike(coord_xyz, volname, label_size_xyz,
                                 label_volume_map)
    # patch shape: [1, Z, Y, X, 1]
    patch = load_from_numpylike(coord_xyz, volname, image_size_xyz,
                                image_volume_map)

    # Calculate radii to find the center voxel
    label_radii = np.array(label_size_xyz) // 2

    # LOM (Logic of Mask): Select ONLY the object present at the center voxel
    # lom shape: [1, Z, Y, X, 1] (Boolean mask)
    lom = np.logical_and(
        labels > 0,
        labels == labels[0,
                         label_radii[2], # Z center
                         label_radii[1], # Y center
                         label_radii[0], # X center
                         0])
    
    # Convert Boolean mask to soft float targets (0.95 and 0.05)
    # labels shape: [1, Z, Y, X, 1] (float32)
    labels = soften_labels(lom)
    
    # Initialize weights for each voxel
    # loss_weights shape: [1, Z, Y, X, 1]
    loss_weights = np.ones_like(labels, dtype=np.float32)

    # Apply spatial data augmentation (Mirroring, Rotation)
    if permutable_axes or reflectable_axes:
        perm_axes = permutable_axes if permutable_axes else []
        refl_axes = reflectable_axes if reflectable_axes else []
        transform = augmentation.PermuteAndReflect(
            rank=5, permutable_axes=perm_axes, reflectable_axes=refl_axes)
        
        # All tensors must undergo the SAME transformation
        labels = transform(labels)         # [1, Z, Y, X, 1]
        patch = transform(patch)           # [1, Z, Y, X, 1]
        loss_weights = transform(loss_weights) # [1, Z, Y, X, 1]

    # Normalize image intensities
    # patch shape: [1, Z, Y, X, 1] (float32)
    patch = offset_and_scale_patches(
        patch, volname,
        offset_scale_map=offset_scale_map,
        default_offset=image_mean,
        default_scale=image_stddev)

    return patch, labels, loss_weights, coord_xyz, volname