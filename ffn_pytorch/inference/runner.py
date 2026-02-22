"""FFN inference runner (PyTorch).

Runner is the top-level orchestrator for inference. It:
1. Initialization: parse config, load model checkpoint, open data volume, start executor
2. Subvolume handling: create Canvas per subvolume, run segmentation, save results
3. Resource management: executor lifecycle, checkpoints, counters

Inference config is passed via protobuf InferenceRequest:
- Model info (name, args, checkpoint path)
- Data volume path (image, optional initial segmentation)
- Inference options (thresholds, min size, etc.)
- Mask config (exclusion regions)

Differences from original TF version:
- TF: tf.Session + tf.train.Saver + session.run()
- PyTorch: torch.load() + model.load_state_dict() + model()
- TF: _init_tf_model() builds graph and restores variables
- PyTorch: _init_pytorch_model() instantiates model and loads weights
- TF: require_gpu checks TF GPU device
- PyTorch: torch.cuda.is_available() checks CUDA

Checkpoint format: PyTorch checkpoint may be any of:
- {'model_state_dict': ...}: standard format from training script
- {'model': ...}: simplified format
- Raw state_dict: minimal format
"""

import copy
import functools
import json
import os
from typing import Any, Optional

from absl import flags
from absl import logging
from ..utils import bounding_box
import numpy as np
import torch

from ..training import model as ffn_model
from ..training.import_util import import_symbol
from . import align
from . import executor
from . import inference
from . import movement
from . import seed
from . import storage
from .inference_utils import timer_counter, Counters

try:
    from ffn.inference import inference_pb2
except (ImportError, TypeError):
    inference_pb2 = None

REQUIRE_GPU = flags.DEFINE_boolean(
    'require_gpu',
    True,
    'Whether to crash in case a GPU device cannot be acquired.',
)

from typing import Tuple as TypingTuple
Tuple3i = TypingTuple[int, int, int]


class Runner:
    """Helper for managing FFN inference runs (PyTorch).

    Takes care of initializing the FFN model and any related functionality
    (e.g. movement policies), as well as input/output of the FFN inference
    data (loading inputs, saving segmentations).
    """

    ALL_MASKED = 1

    def __init__(self):
        self.counters = Counters()
        self.executor = None
        self._exec_interface = executor.ExecutorInterface()
        self.canvases = {}
        self._model = None
        self._device = None

    def __del__(self):
        self.stop_executor()

    def stop_executor(self):
        if self.executor is not None:
            try:
                self.executor.stop_server()
            except executor.TerminationException:
                pass
            self.executor = None

    def _load_pytorch_checkpoint(self, model: torch.nn.Module,
                                  checkpoint_path: str):
        """Loads a PyTorch model checkpoint."""
        with inference.timer_counter(self.counters, 'restore-pytorch-checkpoint'):
            logging.info('Loading PyTorch checkpoint: %s', checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            logging.info('PyTorch checkpoint loaded.')

    def _get_model_class(self, model_name: str):
        return import_symbol(model_name)

    def _init_pytorch_model(self, request, batch_size: int):
        """Initializes a PyTorch model."""
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            if REQUIRE_GPU.value:
                raise RuntimeError('Failed to initialize a GPU device for PyTorch.')
            self._device = torch.device('cpu')

        logging.info('Using device: %s', self._device)

        model_class = self._get_model_class(request.model_name)
        if request.model_args:
            args = json.loads(request.model_args)
        else:
            args = {}
        args['batch_size'] = batch_size
        model = model_class(**args)
        model = model.to(self._device)

        self._model_info = model.info
        self._model = model

        self.executor = executor.ThreadingBatchExecutor(
            self._exec_interface,
            model,
            model.info,
            self.counters,
            batch_size,
        )

        self._load_pytorch_checkpoint(model, request.model_checkpoint_path)

    def start(self, request, batch_size: int = 1):
        """Opens input volumes and initializes the FFN."""
        request = copy.deepcopy(request)
        self.request = request
        assert self.request.segmentation_output_dir
        logging.debug('Received request:\n%s', request)
        print('Received request:\n%s', request)

        os.makedirs(request.segmentation_output_dir, exist_ok=True)

        self.stop_executor()
        self._init_pytorch_model(request, batch_size)

        with timer_counter(self.counters, 'volstore-open'):
            self._image_volume = storage.decorated_volume(request.image)
            assert self._image_volume is not None

            if request.HasField('init_segmentation'):
                self.init_seg_volume = storage.decorated_volume(
                    request.init_segmentation)
            else:
                self.init_seg_volume = None

            def _open_or_none(settings):
                if settings.WhichOneof('volume_path') is None:
                    return None
                return storage.decorated_volume(settings)

            self._mask_volumes = {}
            self._shift_mask_volume = _open_or_none(request.shift_mask)

            alignment_options = request.alignment_options
            if inference_pb2 is not None:
                null_alignment = inference_pb2.AlignmentOptions.NO_ALIGNMENT
            else:
                null_alignment = 0

            if not alignment_options or alignment_options.type == null_alignment:
                self._aligner = align.Aligner()
            else:
                raise NotImplementedError('Non-trivial alignment not implemented.')

        assert self.executor is not None
        self.executor.start_server()

    def make_restrictor(self, corner, subvol_size, image, alignment):
        """Builds a MovementRestrictor object."""
        kwargs = {}

        if self.request.masks:
            with timer_counter(self.counters, 'load-mask'):
                final_mask = storage.build_mask(
                    self.request.masks,
                    corner,
                    subvol_size,
                    self._mask_volumes,
                    image,
                    alignment,
                )
                if np.all(final_mask):
                    logging.info('Everything masked.')
                    return self.ALL_MASKED
                kwargs['mask'] = final_mask

        if self.request.seed_masks:
            with timer_counter(self.counters, 'load-seed-mask'):
                seed_mask = storage.build_mask(
                    self.request.seed_masks,
                    corner,
                    subvol_size,
                    self._mask_volumes,
                    image,
                    alignment,
                )
                if np.all(seed_mask):
                    logging.info('All seeds masked.')
                    return self.ALL_MASKED
                kwargs['seed_mask'] = seed_mask

        if self._shift_mask_volume:
            with timer_counter(self.counters, 'load-shift-mask'):
                s = self.request.shift_mask_scale
                shift_corner = np.array(corner) // (1, s, s)
                shift_size = -(-np.array(subvol_size) // (1, s, s))

                shift_alignment = alignment.rescaled(
                    np.array((1.0, 1.0, 1.0)) / (1, s, s))
                src_corner, src_size = shift_alignment.expand_bounds(
                    shift_corner, shift_size, forward=False)
                src_corner, src_size = storage.clip_subvolume_to_bounds(
                    src_corner, src_size, self._shift_mask_volume)
                src_end = src_corner + src_size

                expanded_shift_mask = self._shift_mask_volume[
                    0:2,
                    src_corner[0]:src_end[0],
                    src_corner[1]:src_end[1],
                    src_corner[2]:src_end[2],
                ]
                shift_mask = np.array([
                    shift_alignment.align_and_crop(
                        src_corner, expanded_shift_mask[i], shift_corner, shift_size)
                    for i in range(2)
                ])
                shift_mask = alignment.transform_shift_mask(corner, s, shift_mask)

                if self.request.HasField('shift_mask_fov'):
                    shift_mask_fov = bounding_box.BoundingBox(
                        start=self.request.shift_mask_fov.start,
                        size=self.request.shift_mask_fov.size,
                    )
                else:
                    shift_mask_diameter = np.array(self._model_info.input_image_size)
                    shift_mask_fov = bounding_box.BoundingBox(
                        start=-(shift_mask_diameter // 2), size=shift_mask_diameter)

                kwargs.update({
                    'shift_mask': shift_mask,
                    'shift_mask_fov': shift_mask_fov,
                    'shift_mask_scale': self.request.shift_mask_scale,
                    'shift_mask_threshold': self.request.shift_mask_threshold,
                })

            return movement.MovementRestrictor(**kwargs) if kwargs else None

    def make_canvas(
        self, corner: Tuple3i, subvol_size: Tuple3i, **canvas_kwargs
    ):
        subvol_counters = self.counters.get_sub_counters()
        with timer_counter(subvol_counters, 'load-image'):
            logging.info('Process subvolume: %r', corner)

            alignment = self._aligner.generate_alignment(corner, subvol_size)

            dst_corner, dst_size = alignment.expand_bounds(
                corner, subvol_size, forward=True)
            src_corner, src_size = alignment.expand_bounds(
                dst_corner, dst_size, forward=False)
            src_corner, src_size = storage.clip_subvolume_to_bounds(
                src_corner, src_size, self._image_volume)

            logging.info('Requested bounds are %r + %r', corner, subvol_size)
            logging.info('Destination bounds are %r + %r', dst_corner, dst_size)
            logging.info('Fetch bounds are %r + %r', src_corner, src_size)

            def get_data_3d(volume, bbox):
                slc = bbox.to_slice3d()
                assert volume is not None
                if volume.ndim == 4:
                    slc = np.index_exp[0:1] + slc
                data = volume[slc]
                if data.ndim == 4:
                    data = data.squeeze(axis=0)
                return data

            src_bbox = bounding_box.BoundingBox(
                start=src_corner[::-1], size=src_size[::-1])
            src_image = get_data_3d(self._image_volume, src_bbox)
            logging.info(
                'Fetched image of size %r prior to transform', src_image.shape)

            def align_and_crop(image):
                return alignment.align_and_crop(
                    src_corner, image, dst_corner, dst_size, forward=True)

            image = align_and_crop(src_image)
            logging.info('Image data loaded, shape: %r.', image.shape)

        restrictor = self.make_restrictor(dst_corner, dst_size, image, alignment)
        if restrictor == self.ALL_MASKED:
            return None, None

        image = (
            image.astype(np.float32) - self.request.image_mean
        ) / self.request.image_stddev

        exc = self.executor
        if exc is None:
            raise executor.TerminationException

        canvas = inference.Canvas(
            self._model_info,
            exc.get_client(subvol_counters),
            image,
            self.request.inference_options,
            counters=subvol_counters,
            restrictor=restrictor,
            movement_policy_fn=movement.get_policy_fn(
                self.request, self._model_info),
            checkpoint_path=storage.checkpoint_path(
                self.request.segmentation_output_dir, corner),
            checkpoint_interval_sec=self.request.checkpoint_interval,
            corner_zyx=dst_corner,
            **canvas_kwargs
        )

        if self.request.HasField('init_segmentation'):
            canvas.init_segmentation_from_volume(
                self.init_seg_volume, src_corner, src_bbox.end[::-1], align_and_crop)
        return canvas, alignment

    def get_seed_policy(self, corner, subvol_size):
        policy_cls = getattr(seed, self.request.seed_policy)
        kwargs = {'corner': corner, 'subvol_size': subvol_size}
        if self.request.seed_policy_args:
            kwargs.update(json.loads(self.request.seed_policy_args))
        return functools.partial(policy_cls, **kwargs)

    def save_segmentation(self, canvas, alignment, target_path, prob_path):
        def unalign_image(im3d):
            if alignment is None:
                return im3d
            return alignment.align_and_crop(
                canvas.corner_zyx,
                im3d,
                alignment.corner,
                alignment.size,
                forward=False,
            )

        def unalign_origins(origins, canvas_corner):
            out_origins = dict()
            for key, value in origins.items():
                zyx = np.array(value.start_zyx) + canvas_corner
                zyx = alignment.transform(zyx[:, np.newaxis], forward=False).squeeze()
                zyx -= canvas_corner
                out_origins[key] = value._replace(start_zyx=tuple(zyx))
            return out_origins

        canvas.segmentation[canvas.segmentation < 0] = 0

        storage.save_subvolume(
            unalign_image(canvas.segmentation),
            unalign_origins(canvas.origins, np.array(canvas.corner_zyx)),
            target_path,
            request=self.request.SerializeToString(),
            counters=canvas.counters.dumps(),
            overlaps=canvas.overlaps,
        )

        if canvas.seg_prob is None:
            print('No seg_prob to save!!!')
        else:
            prob = unalign_image(canvas.seg_prob)
            with storage.atomic_file(prob_path) as fd:
                np.savez_compressed(fd, qprob=prob)

    def run(self, corner: Tuple3i, subvol_size: Tuple3i, reset_counters=True):
        if reset_counters:
            self.counters.reset()

        seg_path = storage.segmentation_path(
            self.request.segmentation_output_dir, corner)
        prob_path = storage.object_prob_path(
            self.request.segmentation_output_dir, corner)
        cpoint_path = storage.checkpoint_path(
            self.request.segmentation_output_dir, corner)
        print('seg_path:', seg_path)
        print('prob_path:', prob_path)
        print('cpoint_path:', cpoint_path)

        if os.path.exists(seg_path):
            print('Segmentation already exists, skipping subvolume.')
            return None

        canvas, alignment = self.make_canvas(corner, subvol_size)
        if canvas is None:
            return None

        assert alignment is not None

        partial_segment_iters = 0
        if os.path.exists(cpoint_path):
            partial_segment_iters = canvas.restore_checkpoint(cpoint_path)

        if self.request.alignment_options.save_raw:
            image_path = storage.subvolume_path(
                self.request.segmentation_output_dir, corner, 'align')
            with storage.atomic_file(image_path) as fd:
                np.savez_compressed(fd, im=canvas.image)

        self.canvases[corner] = canvas
        canvas.segment_all(
            seed_policy=self.get_seed_policy(corner, subvol_size),
            partial_segment_iters=partial_segment_iters,
        )
        self.save_segmentation(canvas, alignment, seg_path, prob_path)
        del self.canvases[corner]

        try:
            os.remove(cpoint_path)
        except OSError:
            pass

        return canvas
