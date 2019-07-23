"@author 1: mm_islam

"@author 2: Tgtadewos

#Supervisor: Dr. Ali Karimoddini

# This modified ROS node provides the specific coordinates, approximate distance, and specific size of the detected pedestrians. By partitioning, the image into three parts the region of interest (left, right, or middle) for each pedestrian is reported. Also, based on the size of the detection window, the approximate distance from a pedestrian in the field of view of the camera is estimated.

# This code can be used with other trained models as well. 

# To validate the developed model, the code has been tested via on-campus and off-campus autonomous driving facilities.

#This effort is supervised by Dr. Ali Karimoddni and supported by STATE OF NORTH CAROLINA, DEPARTMENT OF TRANSPORTATION under the project number 2019-28, led by the Institute for Transportation Research and Education (ITRE) at NC State University and Co-led by Autonomous Cooperative Control of Emergent Systems of Systems (ACCESS) Laboratory at NC A&T State University.

#This code is a modification of multiple object detection with SSD_mobilenet V1.

# The original code is from https://github.com/osrf/tensorflow_object_detector/blob/master/README.md.

# We customized the code to enforce and limit object detection to identify only pedestrians.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================

r"""Tool to export an object detection model for inference.

Prepares an object detection tensorflow graph for inference using model
configuration and an optional trained checkpoint. Outputs either an inference
graph or a SavedModel (https://tensorflow.github.io/serving/serving_basic.html).

The inference graph contains one of three input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [1, None, None, 3]
  * `encoded_image_string_tensor`: Accepts a scalar string tensor of encoded PNG
    or JPEG image.
  * `tf_example`: Accepts a serialized TFExample proto. The batch size in this
    case is always 1.

and the following output nodes returned by the model.postprocess(..):
  * `num_detections`: Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`: Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores`: Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.
  * `detection_masks`: Outputs float32 tensors of the form
      [batch, num_boxes, mask_height, mask_width] containing predicted instance
      masks for each box if its present in the dictionary of postprocessed
      tensors returned by the model.

Note that currently `batch` is always 1, but we will support `batch` > 1 in
the future.

Optionally, one can freeze the graph by converting the weights in the provided
checkpoint as graph constants thereby eliminating the need to use a checkpoint
file during inference.

Note that this tool uses `use_moving_averages` from eval_config to decide
which weights to freeze.

Example Usage:
--------------
python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --checkpoint_path path/to/model-ckpt \
    --inference_graph_path path/to/inference_graph.pb
"""
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('checkpoint_path', '', 'Optional path to checkpoint file. '
                    'If provided, bakes the weights from the checkpoint into '
                    'the graph.')
flags.DEFINE_string('inference_graph_path', '', 'Path to write the output '
                    'inference graph.')
flags.DEFINE_bool('export_as_saved_model', False, 'Whether the exported graph '
                  'should be saved as a SavedModel')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.pipeline_config_path, 'TrainEvalPipelineConfig missing.'
  assert FLAGS.inference_graph_path, 'Inference graph path missing.'
  assert FLAGS.input_type, 'Input type missing.'
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  exporter.export_inference_graph(FLAGS.input_type, pipeline_config,
                                  FLAGS.checkpoint_path,
                                  FLAGS.inference_graph_path,
                                  FLAGS.export_as_saved_model)


if __name__ == '__main__':
  tf.app.run()
