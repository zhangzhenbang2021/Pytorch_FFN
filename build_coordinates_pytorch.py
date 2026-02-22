#!/usr/bin/env python
"""
Generates FFN (Flood-Filling Network) training coordinates.
(Pure PyTorch GPU implementation, ultra-memory-efficient + zero-string disk I/O).

=============================================================================
FINAL OUTPUT FORMAT & SHAPES EXPECTED IN THE HDF5 FILE:
-----------------------------------------------------------------------------
1. Dataset 'coords'      : Shape (N, 3), dtype int32.
                           Each row is a single training point representing
                           the absolute spatial coordinates [X, Y, Z].
2. Dataset 'vol_indices' : Shape (N,), dtype int32.
                           Each element is an integer ID (e.g., 0, 1) indicating
                           which volume the corresponding coordinate came from.
3. Attribute 'label_map' : Array of bytes (UTF-8 encoded strings).
                           Acts as a dictionary mapping the integer ID back
                           to the original volume name (e.g., 0 -> "volume_A").
=============================================================================
"""
from collections import defaultdict
from absl import app, flags, logging
from tqdm import tqdm

import h5py
import numpy as np
import torch
import time

FLAGS = flags.FLAGS

flags.DEFINE_list('partition_volumes', None, 'Partition volumes...')
flags.DEFINE_string('coordinate_output', None, 'Output HDF5 path...')
flags.DEFINE_list('margin', None, 'Margin as (z, y, x)...')
flags.DEFINE_integer('target_count', None, 'Target count per partition for oversampling...')

IGNORE_PARTITION = 255

"""I0221 17:34:07.483453  build_coordinates_pytorch.py:43] Running entirely on device: cuda
    Partition counts:
    0: 1950313
    1: 152169
    2: 2298482
    3: 3771924
    4: 3760250
    5: 11226931
    6: 9552541
    7: 11414567
    8: 13567244
    9: 13263247
    10: 11077441
    11: 8687871
    12: 6896361
    13: 7534707
    Loading and GPU indexing took 4.3s
    Resampling and shuffling coordinates on GPU...
    Max partition count (for oversampling): 13567244
    GPU Resampling took 0.0s, total samples: 189941416
    Converting flat indices back to (x, y, z) coordinates on GPU...
    GPU Coordinate conversion took 0.1s
    Transferring back to CPU and saving to HDF5...
    Final coordinate array shape: (189941416, 3), dtype: int32
"""

def main(argv):
    del argv
    t0 = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Running entirely on device: {device}')

    totals = defaultdict(int)
    indices = defaultdict(list)

    vol_labels = []
    vol_shapes = []
    mz, my, mx = [int(x) for x in FLAGS.margin]

    # ==========================================
    # 1. Ultra-fast Loading & Flattening Phase
    # ==========================================
    for i, partvol in enumerate(FLAGS.partition_volumes):
        name, path, dataset = partvol.split(':')
        with h5py.File(path, 'r') as f:
            # Crop the margins right during the read process to avoid out-of-bounds later.
            part_np = f[dataset][mz:-mz, my:-my, mx:-mx]
            part_t = torch.tensor(part_np, device=device, dtype=torch.uint8)

            vol_shapes.append(part_np.shape)
            vol_labels.append(name) # e.g., "validation1"

            # Find all unique partition IDs and their voxel counts in this volume
            uniques, counts = torch.unique(part_t, return_counts=True)
            uniques_cpu = uniques.cpu().numpy()
            counts_cpu = counts.cpu().numpy()
            print(f'Volume "{name}" unique partitions: {uniques_cpu}, counts: {counts_cpu}')

            # Flatten the 3D volume into a 1D array to speed up index searching
            part_flat = part_t.view(-1)

            for val, cnt in zip(uniques_cpu, counts_cpu):
                if val == IGNORE_PARTITION:
                    continue
                totals[val] += cnt
                
                # Extract 1D flat indices for all voxels belonging to the current partition 'val'.
                # flat_indices Shape: (K,), where K is the number of voxels for this partition.
                flat_indices = torch.nonzero(part_flat == val).squeeze(-1).to(torch.int32)
                
                # Create an array of the same shape filled with the current volume ID 'i'.
                # vol_ids Shape: (K,), elements are all 'i' (e.g., 0, 0, 0...).
                vol_ids = torch.full_like(flat_indices, fill_value=i, dtype=torch.int32)
                
                # Stack them together into pairs.
                # pairs Shape: (K, 2). 
                # Element 0: Volume ID (i). Element 1: 1D Flat Index.
                pairs = torch.stack((vol_ids, flat_indices), dim=1)
                indices[val].append(pairs)
            
            del part_t, part_flat
            torch.cuda.empty_cache()

    logging.info('Partition counts:')
    for k, v in sorted(totals.items()):
        logging.info(' %d: %d', k, v)

    t1 = time.time()
    logging.info('Loading and GPU indexing took %.1fs', t1 - t0)

    # ==========================================
    # 2. VRAM-level Resampling & Shuffling (Oversampling)
    # ==========================================
    logging.info('Resampling and shuffling coordinates on GPU...')
    

    target_count = int(FLAGS.target_count)  # e.g. 10000 or 500000 per partition
    total_classes = len(indices)
    total_samples = total_classes * target_count

    # Pre-allocate memory for the resampled coordinates.
    # resampled Shape: (total_samples, 2).
    # Column 0: Volume ID. Column 1: 1D Flat Index.
    resampled = torch.empty((total_samples, 2), dtype=torch.int32, device=device)
    
    current_row = 0
    for val, v_list in tqdm(indices.items()):
        v_tensor = torch.cat(v_list, dim=0) 
        N = len(v_tensor)
        
        # Shuffle the current partition's coordinates
        perm = torch.randperm(N, device=device)
        shuffled = v_tensor[perm]
        
        # Clever modulo trick to repeat the shuffled array until it reaches 'target_count'
        idx = torch.arange(target_count, device=device) % N
        
        resampled[current_row : current_row + target_count] = shuffled[idx]
        current_row += target_count
        
        del v_tensor, perm, shuffled, idx

    # Perform a global shuffle across all balanced classes
    global_perm = torch.randperm(total_samples, device=device)
    resampled = resampled[global_perm]
    del global_perm

    t2 = time.time()
    logging.info('GPU Resampling took %.1fs, total samples: %d', t2 - t1, len(resampled))

    # ==========================================
    # 3. GPU Vectorized Coordinate Restoration
    # ==========================================
    logging.info('Converting flat indices back to (x, y, z) coordinates on GPU...')

    vol_indices = resampled[:, 0]
    flat_coords = resampled[:, 1]

    # Pre-allocate tensor for final 3D coordinates.
    # coords_tensor Shape: (total_samples, 3).
    # Column 0: X coordinate. Column 1: Y coordinate. Column 2: Z coordinate.
    coords_tensor = torch.empty((total_samples, 3), dtype=torch.int32, device=device)
    
    for vi in range(len(vol_labels)):
        mask = (vol_indices == vi)
        if not mask.any():
            continue

        shape = vol_shapes[vi]
        flat_sub = flat_coords[mask]

        S_y, S_x = shape[1], shape[2]
        
        # Mathematical equivalent of unravel_index (reversing the flattening process)
        z = flat_sub // (S_y * S_x)
        rem = flat_sub % (S_y * S_x)
        y = rem // S_x
        x = rem % S_x

        # Add the margins back to get the absolute coordinates in the original uncropped volume.
        coords_tensor[mask, 0] = x + mx
        coords_tensor[mask, 1] = y + my
        coords_tensor[mask, 2] = z + mz

    t3 = time.time()
    logging.info('GPU Coordinate conversion took %.1fs', t3 - t2)

    # ==========================================
    # 4. Transfer to CPU & Fast HDF5 Writing
    # ==========================================
    logging.info('Transferring back to CPU and saving to HDF5...')
    
    coords_array = coords_tensor.cpu().numpy()
    vol_indices_cpu = vol_indices.cpu().numpy()
    
    # Confirming the final shape: Expected (N, 3), dtype int32
    print(f'Final coordinate array shape: {coords_array.shape}, dtype: {coords_array.dtype}')
    
    del coords_tensor, resampled, vol_indices, flat_coords
    torch.cuda.empty_cache()
    
    # CORE MODIFICATION: Save unique labels as a dictionary/map instead of repeating strings.
    # label_map_ascii Shape: (Number of volumes,). Array of UTF-8 byte strings.
    label_map_ascii = np.array([lbl.encode('utf-8') for lbl in vol_labels])
    
    with h5py.File(FLAGS.coordinate_output, 'w') as f:
        # Save 3D coordinates [X, Y, Z]
        f.create_dataset('coords', data=coords_array,
                         compression='gzip', compression_opts=4)
        
        # Save Volume IDs (Integers like 0, 1). Massive memory savings compared to strings!
        f.create_dataset('vol_indices', data=vol_indices_cpu,
                         compression='gzip', compression_opts=4)
        
        # Mount the string mapping as a global attribute on the HDF5 file
        f.attrs['label_map'] = label_map_ascii

    t4 = time.time()
    logging.info('HDF5 write took %.1fs', t4 - t3)
    logging.info('Done. Total time: %.1fs. Saved %d coordinates.',
                 t4 - t0, len(coords_array))

if __name__ == '__main__':
    flags.mark_flag_as_required('margin')
    flags.mark_flag_as_required('coordinate_output')
    flags.mark_flag_as_required('partition_volumes')
    flags.mark_flag_as_required('target_count')
    app.run(main)
    

# =============================================================================
# OPTIMIZED PYTORCH DATALOADER IMPLEMENTATION
# =============================================================================

# import torch
# from torch.utils.data import Dataset, DataLoader
# import h5py

# class FFNTrainingDataset(Dataset):
#     def __init__(self, hdf5_path):
#         super().__init__()
#         self.hdf5_path = hdf5_path
        
#         # 1. Open the HDF5 file in read mode
#         self.h5_file = h5py.File(self.hdf5_path, 'r')
        
#         # 2. Get references to the datasets (LAZY LOADING).
#         # IMPORTANT: This does NOT load the 100M+ coordinates into RAM!
#         # It only creates a pointer to the disk data, keeping RAM usage near zero.
#         self.coords = self.h5_file['coords']             # Shape: (N, 3) on disk
#         self.vol_indices = self.h5_file['vol_indices']   # Shape: (N,) on disk
#         self.length = self.coords.shape[0]               # N (Total samples)
        
#         # 3. Read the dictionary map and decode bytes back to regular Python strings.
#         # Result example: ['validation1', 'validation2']
#         raw_labels = self.h5_file.attrs['label_map']
#         self.label_map = [lbl.decode('utf-8') for lbl in raw_labels]

#     def __len__(self):
#         # Returns the total number of training coordinates (N)
#         return self.length

#     def __getitem__(self, idx):
#         """
#         Retrieves a single training coordinate and its corresponding volume name.
#         """
#         # Fetch the specific coordinate [X, Y, Z] and its volume integer ID.
#         # Note: Minor disk I/O happens here, but DataLoader multiprocessing masks the latency.
#         coord = self.coords[idx]       # Array of shape (3,), e.g., [150, 200, 50]
#         vol_idx = self.vol_indices[idx] # Integer, e.g., 0
        
#         # [MAGIC RESTORATION]: Look up the actual string name using the integer ID.
#         label_name = self.label_map[vol_idx] # e.g., "validation1"
        
#         # Returns:
#         # 1. Tensor of shape (3,) representing [X, Y, Z], dtype int64.
#         # 2. String representing the volume name.
#         return torch.tensor(coord, dtype=torch.int64), label_name
        
#     def __del__(self):
#         # Ensure the HDF5 file is safely closed when the dataset object is destroyed.
#         if hasattr(self, 'h5_file'):
#             self.h5_file.close()

# # ==========================================
# # TEST SCRIPT (You can run this directly to verify)
# # ==========================================
# if __name__ == '__main__':
#     # Assuming your generated file is named validation_coords.h5
#     dataset = FFNTrainingDataset('third_party/neuroproof_examples/validation_sample/validation_coords.h5')
    
#     print(f"Total samples in dataset: {len(dataset)}")
#     print(f"Label Mapping Dictionary: {dataset.label_map}")
    
#     # Randomly inspect the 100,000th sample and the last sample
#     print("\nSample 100,000:", dataset[100000])
#     print("Last sample:", dataset[len(dataset) - 1])
    
#     # Test loading speed with PyTorch DataLoader (Multi-threading enabled)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
#     batch_coords, batch_labels = next(iter(dataloader))
    
#     # batch_coords Shape: (4, 3) because batch_size=4.
#     # batch_labels is a tuple of 4 strings.
#     print("\nBatch Coordinates (Shape: {}):\n{}".format(batch_coords.shape, batch_coords))
#     print("Batch Labels:", batch_labels)