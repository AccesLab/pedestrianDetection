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

"""Tests for box_coder_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.builders import box_coder_builder
from object_detection.protos import box_coder_pb2


class BoxCoderBuilderTest(tf.test.TestCase):

  def test_build_faster_rcnn_box_coder_with_defaults(self):
    box_coder_text_proto = """
      faster_rcnn_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(isinstance(box_coder_object,
                               faster_rcnn_box_coder.FasterRcnnBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0, 5.0])

  def test_build_faster_rcnn_box_coder_with_non_default_parameters(self):
    box_coder_text_proto = """
      faster_rcnn_box_coder {
        y_scale: 6.0
        x_scale: 3.0
        height_scale: 7.0
        width_scale: 8.0
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(isinstance(box_coder_object,
                               faster_rcnn_box_coder.FasterRcnnBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0, 8.0])

  def test_build_mean_stddev_box_coder(self):
    box_coder_text_proto = """
      mean_stddev_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object,
                   mean_stddev_box_coder.MeanStddevBoxCoder))

  def test_build_square_box_coder_with_defaults(self):
    box_coder_text_proto = """
      square_box_coder {
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0])

  def test_build_square_box_coder_with_non_default_parameters(self):
    box_coder_text_proto = """
      square_box_coder {
        y_scale: 6.0
        x_scale: 3.0
        length_scale: 7.0
      }
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    box_coder_object = box_coder_builder.build(box_coder_proto)
    self.assertTrue(
        isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
    self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0])

  def test_raise_error_on_empty_box_coder(self):
    box_coder_text_proto = """
    """
    box_coder_proto = box_coder_pb2.BoxCoder()
    text_format.Merge(box_coder_text_proto, box_coder_proto)
    with self.assertRaises(ValueError):
      box_coder_builder.build(box_coder_proto)


if __name__ == '__main__':
  tf.test.main()
