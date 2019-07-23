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

"""Tests for post_processing_builder."""

import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import post_processing_builder
from object_detection.protos import post_processing_pb2


class PostProcessingBuilderTest(tf.test.TestCase):

  def test_build_non_max_suppressor_with_correct_parameters(self):
    post_processing_text_proto = """
      batch_non_max_suppression {
        score_threshold: 0.7
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
      }
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    non_max_suppressor, _ = post_processing_builder.build(
        post_processing_config)
    self.assertEqual(non_max_suppressor.keywords['max_size_per_class'], 100)
    self.assertEqual(non_max_suppressor.keywords['max_total_size'], 300)
    self.assertAlmostEqual(non_max_suppressor.keywords['score_thresh'], 0.7)
    self.assertAlmostEqual(non_max_suppressor.keywords['iou_thresh'], 0.6)

  def test_build_identity_score_converter(self):
    post_processing_text_proto = """
      score_converter: IDENTITY
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter, tf.identity)

  def test_build_sigmoid_score_converter(self):
    post_processing_text_proto = """
      score_converter: SIGMOID
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter, tf.sigmoid)

  def test_build_softmax_score_converter(self):
    post_processing_text_proto = """
      score_converter: SOFTMAX
    """
    post_processing_config = post_processing_pb2.PostProcessing()
    text_format.Merge(post_processing_text_proto, post_processing_config)
    _, score_converter = post_processing_builder.build(post_processing_config)
    self.assertEqual(score_converter, tf.nn.softmax)


if __name__ == '__main__':
  tf.test.main()
