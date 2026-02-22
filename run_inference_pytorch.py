#!/usr/bin/env python
"""FFN inference entry script (PyTorch).

Runs FFN inference within a 3D bounding box to produce instance segmentation.
Inference runs in a single process.

This script is the entry point for inference. It:
1. Parses InferenceRequest protobuf (model, data, options)
2. Parses BoundingBox protobuf (segmentation region)
3. Initializes Runner (load model, open volume, start executor)
4. Runs segmentation in the given region
5. Saves segmentation results and counters

Inference config is passed via --inference_request in protobuf text format.

Note: For protobuf compatibility, this script sets
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python automatically.

Usage example:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \\
    python run_inference_pytorch.py \\
        --inference_request "
            image { hdf5: 'grayscale.h5:raw' }
            image_mean: 128
            image_stddev: 33
            model_checkpoint_path: 'model-400000.pth'
            model_name: 'convstack_3d.ConvStack3DFFNModel'
            model_args: '{\"depth\":9,\"fov_size\":[33,33,33],\"deltas\":[8,8,8]}'
            segmentation_output_dir: '/tmp/ffn_output'
            inference_options {
                init_activation: 0.95
                pad_value: 0.05
                move_threshold: 0.9
                segment_threshold: 0.6
                min_segment_size: 1000
                min_boundary_dist { x:1 y:1 z:1 }
            }
        " \\
        --bounding_box "start { x:0 y:0 z:0 } size { x:512 y:512 z:256 }"
"""

import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from google.protobuf import text_format
from absl import app
from absl import flags

from ffn.utils import bounding_box_pb2
from ffn_pytorch.inference import inference_flags
from ffn_pytorch.inference import runner as runner_mod

FLAGS = flags.FLAGS

flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segment.')


def main(unused_argv):
    request = inference_flags.request_from_flags()

    os.makedirs(request.segmentation_output_dir, exist_ok=True)

    bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(FLAGS.bounding_box, bbox)

    runner = runner_mod.Runner()
    runner.start(request)
    runner.run((bbox.start.z, bbox.start.y, bbox.start.x),
               (bbox.size.z, bbox.size.y, bbox.size.x))

    counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
    if not os.path.exists(counter_path):
        runner.counters.dump(counter_path)

    print('ALL DONE')


if __name__ == '__main__':
    app.run(main)
