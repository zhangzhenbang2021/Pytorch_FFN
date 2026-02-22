#!/usr/bin/env python
"""FFN training driver script (PyTorch epoch-driven).

This script implements the full FFN training pipeline:
1. Parses HDF5 coordinate files to compute exact steps per epoch.
2. Each epoch trains exactly one full pass over the data.
3. Prints average loss and saves checkpoint only at epoch end.
4. Epoch-based checkpoint resume support.

Usage example:
    nohup python train_pytorch.py \
        --train_coords "coordinates.h5" \
        --data_volumes "mydata:grayscale.h5:raw" \
        --label_volumes "mydata:groundtruth.h5:stack" \
        --model_name "convstack_3d.ConvStack3DFFNModel" \
        --model_args '{"depth":9, "fov_size":[33,33,33], "deltas":[8,8,8]}' \
        --image_mean 128 --image_stddev 33 \
        --train_dir ./tmp/ffn_model \
        --epochs 10 \
        --lr_decay_steps 50000 --lr_decay_factor 0.9 > full_train.log 2>&1 &
"""

from functools import partial
import datetime
import glob
import json
import logging
import os
import random
import time
from absl import app
from absl import flags
import h5py
import numpy as np
from scipy import special
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from ffn_pytorch.training import augmentation
from ffn_pytorch.training import examples
from ffn_pytorch.training import inputs
from ffn_pytorch.training import model as ffn_model
from ffn_pytorch.training import optimizer as opt_util
from ffn_pytorch.training import tracker
from ffn_pytorch.training.import_util import import_symbol

FLAGS = flags.FLAGS

flags.DEFINE_string('train_coords', None,
                    'Glob for coordinate files (TFRecord or HDF5).')
flags.DEFINE_string('data_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:<dataset>.')
flags.DEFINE_string('label_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:<dataset>.')
flags.DEFINE_string('model_name', None,
                    'Name of the model to train.')
flags.DEFINE_string('model_args', None,
                    'JSON string with model constructor arguments.')
flags.DEFINE_string('train_dir', '/tmp/ffn_train',
                    'Path for checkpoints and logs.')

flags.DEFINE_float('seed_pad', 0.05, 'Value for unknown seed area.')
flags.DEFINE_float('threshold', 0.9, 'FOV movement threshold.')
flags.DEFINE_enum('fov_policy', 'fixed', ['fixed', 'max_pred_moves', 'no_step'],
                  'Policy for FOV movement during training.')
flags.DEFINE_integer('fov_moves', 1, 'Number of FOV moves per dimension.')
flags.DEFINE_boolean('shuffle_moves', True, 'Whether to randomize move order.')

flags.DEFINE_float('image_mean', None, 'Mean image intensity for normalization.')
flags.DEFINE_float('image_stddev', None, 'Image stddev for normalization.')
flags.DEFINE_list('image_offset_scale_map', None,
                  'Optional per-volume offset and scale specification.')

flags.DEFINE_list('permutable_axes', ['1', '2'],
                  'Axes that may be permuted for augmentation.')
flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                  'Axes that may be reflected for augmentation.')
flags.DEFINE_float('max_gradient_entry_mag', 0.7,
                   'Max gradient magnitude for clipping.')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('lr_decay_factor', None,
                   'Multiplicative factor by which to decay the learning rate.')
flags.DEFINE_integer('lr_decay_epochs', None,
                     'Number of epochs between learning rate decays.')
flags.DEFINE_integer('batch_size', 4, 'Number of images in a batch.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('checkpoint_interval_epochs', 1, 
                     'How many epochs between saving checkpoints.')
flags.DEFINE_integer('summary_rate_steps', 500, 'Steps between logging to TensorBoard.')

def fov_moves():
    if FLAGS.fov_policy == 'max_pred_moves':
        return FLAGS.fov_moves + 1
    return FLAGS.fov_moves


def train_labels_size(info):
    return (np.array(info.pred_mask_size) +
            np.array(info.deltas) * 2 * fov_moves())


def train_eval_size(info):
    return (np.array(info.pred_mask_size) +
            np.array(info.deltas) * 2 * FLAGS.fov_moves)


def train_image_size(info):
    return (np.array(info.input_image_size) +
            np.array(info.deltas) * 2 * fov_moves())


def train_canvas_size(info):
    return (np.array(info.input_seed_size) +
            np.array(info.deltas) * 2 * fov_moves())


def _get_offset_and_scale_map():
    if not FLAGS.image_offset_scale_map:
        return {}
    ret = {}
    for vol_def in FLAGS.image_offset_scale_map:
        vol_name, offset, scale = vol_def.split(':')
        ret[vol_name] = float(offset), float(scale)
    return ret


def _get_reflectable_axes():
    return [int(x) + 1 for x in FLAGS.reflectable_axes]


def _get_permutable_axes():
    return [int(x) + 1 for x in FLAGS.permutable_axes]


def train_ffn(model_cls, **model_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = model_cls(**model_kwargs).to(device)
    info = model.info
    eval_shape_zyx = train_eval_size(info).tolist()[::-1] # [49,49,49]

    eval_tracker_obj = tracker.EvalTracker(eval_shape_zyx, model.shifts)

    label_volume_map = {}
    for vol in FLAGS.label_volumes.split(','):
        volname, path, dataset = vol.split(':')
        label_volume_map[volname] = h5py.File(path, 'r')[dataset]

    image_volume_map = {}
    for vol in FLAGS.data_volumes.split(','):
        volname, path, dataset = vol.split(':')
        image_volume_map[volname] = h5py.File(path, 'r')[dataset]

    label_size = train_labels_size(info) # [49,49,49]
    image_size = train_image_size(info) # [49,49,49]

    # ================= Parse HDF5 coords to get steps per epoch =================
    total_samples = 0
    for file_path in glob.glob(FLAGS.train_coords):
        with h5py.File(file_path, 'r') as f:
            total_samples += len(f['coords'])
            
    steps_per_epoch = total_samples // FLAGS.batch_size
    logging.info(f"==> automatically calculated total training samples: {total_samples}")
    logging.info(f"==> batch size: {FLAGS.batch_size}, steps per epoch: {steps_per_epoch}")
    
    # =================================================================

    coord_loader = inputs.CoordinateLoader(FLAGS.train_coords, shuffle=True)

    optim = opt_util.optimizer_from_args(
        model.parameters(),
        optimizer_name=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate)
    
    # Compute decay steps: e.g. decay every 2 epochs => 2 * steps_per_epoch
    actual_decay_steps = None
    if getattr(FLAGS, 'lr_decay_epochs', None) is not None:
        actual_decay_steps = FLAGS.lr_decay_epochs * steps_per_epoch

    scheduler = opt_util.build_lr_scheduler(
        optim, 
        decay_factor=getattr(FLAGS, 'lr_decay_factor', None), 
        decay_steps=actual_decay_steps  # total steps for scheduler
    )

    os.makedirs(FLAGS.train_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=FLAGS.train_dir)

    fov_shifts = list(model.shifts)
    if FLAGS.shuffle_moves:
        random.shuffle(fov_shifts)

    train_image_radius = train_image_size(info) // 2 # [24,24,24]
    input_image_radius = np.array(info.input_image_size) // 2 # [16,16,16]
    policy_map = {
        'fixed': partial(
            examples.fixed_offsets,
            fov_shifts=fov_shifts,
            threshold=special.logit(FLAGS.threshold)),
        'max_pred_moves': partial(
            examples.max_pred_offsets,
            max_radius=train_image_radius - input_image_radius,
            threshold=special.logit(FLAGS.threshold)),
        'no_step': examples.no_offsets,
    }
    policy_fn = policy_map[FLAGS.fov_policy]

    def _load_example():
        return inputs.load_example(
            coord_loader, label_volume_map, image_volume_map,
            label_size.tolist(), image_size.tolist(),
            FLAGS.image_mean, FLAGS.image_stddev,
            offset_scale_map=_get_offset_and_scale_map(),
            permutable_axes=_get_permutable_axes(),
            reflectable_axes=_get_reflectable_axes())

    def _make_ffn_example():
        return examples.get_example(
            _load_example,
            eval_tracker_obj,
            info,
            policy_fn,
            FLAGS.seed_pad,
            seed_shape=tuple(train_canvas_size(info).tolist()[::-1]))

    batch_it = examples.BatchExampleIter(
        _make_ffn_example, eval_tracker_obj, FLAGS.batch_size, info)

    # ================= Epoch-based checkpoint resume =================
    start_epoch = 0
    global_step = 0
    checkpoints = glob.glob(os.path.join(FLAGS.train_dir, 'model-epoch*.pth'))
    
    if checkpoints:
        # Sort by epoch number
        get_epoch = lambda x: int(os.path.basename(x).replace('model-epoch', '').replace('.pth', ''))
        latest_ckpt = max(checkpoints, key=get_epoch)
        
        logging.info(f"==> Found checkpoint, loading: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logging.info(f"==> Resumed from epoch {start_epoch}.")
    else:
        logging.info("==> No checkpoint found, starting from scratch.")

    model.train()

    logging.info(f'Starting training for {FLAGS.epochs} epochs ({FLAGS.epochs * steps_per_epoch} total steps)...')

    # Init global time and step for ETA
    last_log_time = time.time()
    last_log_step = global_step
    total_steps = FLAGS.epochs * steps_per_epoch

    logging.info(f'Starting training for {FLAGS.epochs} epochs ({total_steps} total steps)...')

    # ================= Train =================
    for epoch in range(start_epoch, FLAGS.epochs):
        logging.info(f"========== Epoch {epoch + 1}/{FLAGS.epochs} ==========")
        
        eval_tracker_obj.reset()
        epoch_loss_sum = 0.0
        
        for step in range(steps_per_epoch):
            # Get data from batch_it; raw format is NumPy (N, D, H, W, C)
            # N: Batch, D: Depth(Z), H: Height(Y), W: Width(X), C: Channels(1)
            # e.g. (4, 33, 33, 33, 1)
            seed_np, patches_np, labels_np, weights_np = next(batch_it)

            # --- Transpose: (N, D, H, W, C) -> (N, C, D, H, W) ---
            # PyTorch 3D conv expects channels first
            # Result shape: (N, 1, 33, 33, 33)
            seed_t = torch.from_numpy(seed_np.transpose(0, 4, 1, 2, 3)).pin_memory().to(device, non_blocking=True)
            patches_t = torch.from_numpy(patches_np.transpose(0, 4, 1, 2, 3)).pin_memory().to(device, non_blocking=True)
            labels_t = torch.from_numpy(labels_np.transpose(0, 4, 1, 2, 3)).pin_memory().to(device, non_blocking=True)
            weights_t = torch.from_numpy(weights_np.transpose(0, 4, 1, 2, 3)).pin_memory().to(device, non_blocking=True)

            # Forward pass
            # logits shape: (N, 1, 33, 33, 33)
            logits = model(patches_t, seed_t)
            
            # Compute loss (scalar)
            loss = model.compute_loss(logits, labels_t, weights_t)

            optim.zero_grad()
            loss.backward()

            if FLAGS.max_gradient_entry_mag > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), FLAGS.max_gradient_entry_mag)
            
            optim.step()
            
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_loss_sum += loss.item()

            # --- Closed-loop feedback: (N, C, D, H, W) -> (N, D, H, W, C) ---
            # Convert predictions back to NumPy and restore layout for batch_it canvas
            # logits.detach() shape: (N, 1, 33, 33, 33); updated_seed_np shape: (N, 33, 33, 33, 1)
            updated_seed_np = logits.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
            
            # Write updated local seeds back to global canvas
            batch_it.update_seeds(updated_seed_np)
            
            writer.add_scalar('Step/pixel_loss', loss.item(), global_step)

            # ================= Log =================
            if global_step > 0 and global_step % FLAGS.summary_rate_steps == 0:
                logging.info('Epoch [%d/%d] Step [%d/%d] (Global %d) | loss=%.4f | %.3f s/step', 
                             epoch + 1, FLAGS.epochs, step + 1, steps_per_epoch, global_step, loss.item())

        # Epoch summary
        avg_loss = epoch_loss_sum / steps_per_epoch
        logging.info(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
    
    
def main(argv=()):
    del argv
    model_class = import_symbol(FLAGS.model_name)
    seed = int(time.time())
    logging.info('Random seed: %r', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ffn(model_class, batch_size=FLAGS.batch_size,
              **json.loads(FLAGS.model_args))


if __name__ == '__main__':
    flags.mark_flag_as_required('train_coords')
    flags.mark_flag_as_required('data_volumes')
    flags.mark_flag_as_required('label_volumes')
    flags.mark_flag_as_required('model_name')
    flags.mark_flag_as_required('model_args')
    app.run(main)