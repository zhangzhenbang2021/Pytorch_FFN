"""Helpers to initialize protos from flag values (PyTorch version)."""

import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from google.protobuf import text_format
from absl import flags

from ffn.inference import inference_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('inference_request', None,
                    'InferenceRequest proto in text format.')
flags.DEFINE_string('inference_options', None,
                    'InferenceOptions proto in text format.')


def options_from_flags():
    options = inference_pb2.InferenceOptions()
    if FLAGS.inference_options:
        text_format.Parse(FLAGS.inference_options, options)
    return options


def request_from_flags():
    request = inference_pb2.InferenceRequest()
    if FLAGS.inference_request:
        text_format.Parse(FLAGS.inference_request, request)
    return request
