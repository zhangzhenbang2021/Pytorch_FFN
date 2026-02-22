# FFN-PyTorch: Flood-Filling Networks (PyTorch Implementation)

This project is a complete PyTorch reimplementation of **Flood-Filling Networks (FFN)**, based on the original paper ([Flood-Filling Networks](https://arxiv.org/abs/1611.00421)) and the official reference code released by Google ([google/ffn](https://github.com/google/ffn/tree/master)).

FFN is a powerful **iterative neural network approach for volumetric instance segmentation**, and it has become a cornerstone technique in connectomics for neuron reconstruction. Although the method was introduced in 2018, it remains highly effective and widely adopted today.

Despite FFN’s strength and continued relevance, relatively few works have built substantial algorithmic improvements on top of it. A key reason, in my view, is the practical barrier imposed by the original TensorFlow 1.x codebase: it is large, tightly coupled to legacy TensorFlow abstractions, and difficult to modify or extend. For many researchers, the engineering cost of navigating and modernizing that framework becomes a bottleneck—slowing down experimentation and limiting the evolution of FFN.

To remove this barrier and help FFN thrive in modern research workflows, I rewrote FFN in **PyTorch**, with an emphasis on readability, modularity, and extensibility. In addition, I **GPU-accelerated the training data generation pipeline**, which is one of the most time-consuming components in the original implementation, enabling faster iteration and more practical large-scale training.

If you use this project in your research, please cite:
- the original FFN paper,
- the Google FFN implementation,
- and this PyTorch reimplementation.

If you plan to train and evaluate models with this codebase, I strongly recommend reading the Google FFN documentation on **data preparation and preprocessing**, which this project intentionally does not replicate in full.

The goal of this project is to lower the engineering barrier and help FFN regain momentum—making it easier for the community to extend, improve, and build new ideas on top of a proven method.

---

## Table of Contents

- [FFN-PyTorch: Flood-Filling Networks (PyTorch Implementation)](#ffn-pytorch-flood-filling-networks-pytorch-implementation)
  - [Table of Contents](#table-of-contents)
  - [Algorithm Overview](#algorithm-overview)
    - [Training Process](#training-process)
  - [Project Structure](#project-structure)
  - [Environment Setup](#environment-setup)
    - [Dependencies](#dependencies)
    - [Conda Environment](#conda-environment)
  - [Data Preparation](#data-preparation)
    - [Training Data Format](#training-data-format)
    - [Compute Partition Volume](#compute-partition-volume)
    - [Generate Training Coordinates](#generate-training-coordinates)
  - [Training Workflow](#training-workflow)
  - [Inference Workflow](#inference-workflow)
  - [Differences from the Original TF Version](#differences-from-the-original-tf-version)

---

## Algorithm Overview

The core idea of FFN is to segment each neuron instance in a 3D volume through an **iterative filling process**:

1. **Seed Initialization**: Place an initial seed at a specific location.
2. **Prediction Expansion**: The neural network predicts which voxels within the current Field of View (FOV) belong to the same object.
3. **FOV Movement**: Move the FOV to neighboring locations based on prediction results.
4. **Iterative Refinement**: Repeat movement and prediction until the object is fully segmented.
5. **Next Object**: Use a seed policy to find the next unsegmented object and repeat steps 1–4.

```

┌────────────────────────────────────────────────────────┐
│                   FFN Inference Flow                  │
│                                                        │
│  Seed Policy ──→ Initial Position ──→ Network Predict │
│      ↑                        │            │           │
│      │                        ▼            ▼           │
│      │                   Update Logits    Generate Next│
│      │                        │           Position     │
│      │                        ▼            ▼           │
│      │                   Threshold Check ← Continue?   │
│      │                        │                        │
│      │                        ▼                        │
│      └────────────────── Next Object                   │
└────────────────────────────────────────────────────────┘

```

### Training Process

During training, the network learns to predict the correct object mask given an image patch and a seed mask:

- **Input**: Image patch `[B, 1, Z, Y, X]` + seed mask `[B, 1, Z, Y, X]` (logit space)
- **Output**: Updated logit mask `[B, 1, Z, Y, X]`
- **Loss**: Weighted sigmoid cross-entropy loss
- **FOV Movement Simulation**: Simulates inference-time FOV movement during training

---

## Project Structure

```

ffn_pytorch/
├── **init**.py
├── README.md                          # This document
│
├── training/                          # ===== Training Module =====
│   ├── **init**.py
│   ├── README.md                      # Training documentation
│   ├── model.py                       # FFNModel base class + ModelInfo dataclass
│   ├── models/                        # Model implementations
│   │   ├── **init**.py
│   │   └── convstack_3d.py            # ConvStack3DFFNModel (3D residual conv stack)
│   ├── inputs.py                      # Data loading (HDF5/TFRecord → NumPy)
│   ├── examples.py                    # Training sample generation + FOV simulation
│   ├── augmentation.py                # Data augmentation
│   ├── mask.py                        # Mask utilities
│   ├── tracker.py                     # Training metrics tracking
│   ├── optimizer.py                   # Optimizer configuration
│   └── import_util.py                 # Dynamic model import
│
├── inference/                         # ===== Inference Module =====
│   ├── **init**.py
│   ├── README.md
│   ├── executor.py                    # Batch inference executor
│   ├── inference.py                   # Canvas state manager
│   ├── runner.py                      # Inference orchestration
│   ├── inference_flags.py             # CLI → protobuf parsing
│   ├── inference_utils.py             # Utility functions
│   ├── movement.py                    # FOV movement policy
│   ├── seed.py                        # Seed policies
│   ├── segmentation.py                # Post-processing
│   ├── storage.py                     # I/O
│   └── align.py                       # Alignment tools
│
└── utils/                             # ===== Utilities =====
├── **init**.py
└── bounding_box.py

````

---

## Environment Setup

### Dependencies

```bash
pip install torch torchvision
pip install numpy scipy scikit-image
pip install h5py
pip install Pillow
pip install absl-py
pip install protobuf
pip install tensorboard
````

Optional:

```bash
pip install tensorstore
pip install connectomics
```

### Conda Environment

```bash
conda activate torch
```

---

## Data Preparation

### Training Data Format

* **Image Volume**: `[Z, Y, X]` or `[C, Z, Y, X]`
* **Label Volume**: `[Z, Y, X]` (integer labels, 0 = background)

### Compute Partition Volume

```bash
python compute_partitions_pytorch.py \
    --input_volume groundtruth.h5:stack \
    --output_volume af.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 16,16,16 \
    --min_size 10000
```

### Generate Training Coordinates

```bash
python build_coordinates_pytorch.py \
    --partition_volumes validation1:third_party/neuroproof_examples/validation_sample/af.h5:af \
    --coordinate_output third_party/neuroproof_examples/validation_sample/validation_coords.h5 \
    --margin 24,24,24 --target_count 10000
```

---

## Training Workflow

```bash
python train_pytorch.py \
    --train_coords "third_party/neuroproof_examples/validation_sample/validation_coords.h5" \
    --data_volumes "validation1:third_party/neuroproof_examples/validation_sample/grayscale_maps.h5:raw" \
    --label_volumes "validation1:third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack" \
    --model_name "convstack_3d.ConvStack3DFFNModel" \
    --model_args '{"depth": 9, "fov_size": [33,33,33], "deltas": [8,8,8]}' \
    --image_mean 128 --image_stddev 33 \
    --train_dir ./tmp/ffn_model2 \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --lr_decay_epochs 25 --lr_decay_factor 0.9 \
    --summary_rate_steps 500 --checkpoint_interval_epochs 10 
```

---

## Inference Workflow

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python run_inference_pytorch.py \
    --inference_request "
        image { hdf5: 'third_party/neuroproof_examples/training_sample2/grayscale_maps.h5:raw' }
        image_mean: 128
        image_stddev: 33
        model_checkpoint_path: './tmp/ffn_model/model-epoch100.pth'
        model_name: 'convstack_3d.ConvStack3DFFNModel'
        model_args: '{\"depth\": 9, \"fov_size\": [33,33,33], \"deltas\": [8,8,8]}'
        segmentation_output_dir: './tmp/ffn_output'
        seed_policy: 'PolicyPeaks'
        inference_options {
            init_activation: 0.95
            pad_value: 0.05
            move_threshold: 0.9
            segment_threshold: 0.6
            min_segment_size: 1000
            min_boundary_dist { x:1 y:1 z:1 }
        }
    " \
    --bounding_box "start { x: 0 y: 0 z: 0 } size { x: 250 y: 250 z: 250 }"

```

---

## Differences from the Original TF Version

| Component   | TensorFlow             | PyTorch          |
| ----------- | ---------------------- | ---------------- |
| Model       | tf_slim                | nn.Module        |
| Execution   | session.run            | model(x)         |
| Optimizer   | tf.train.AdamOptimizer | torch.optim.Adam |
| Checkpoints | TF checkpoint          | .pth             |
| Data Input  | TFRecord               | HDF5             |


