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

"""Tests for object_detection.builders.image_resizer_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import image_resizer_builder
from object_detection.protos import image_resizer_pb2


class ImageResizerBuilderTest(tf.test.TestCase):

  def _shape_of_resized_random_image_given_text_proto(
      self, input_shape, text_proto):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    text_format.Merge(text_proto, image_resizer_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    images = tf.to_float(tf.random_uniform(
        input_shape, minval=0, maxval=255, dtype=tf.int32))
    resized_images = image_resizer_fn(images)
    with self.test_session() as sess:
      return sess.run(resized_images).shape

  def test_built_keep_aspect_ratio_resizer_returns_expected_shape(self):
    image_resizer_text_proto = """
      keep_aspect_ratio_resizer {
        min_dimension: 10
        max_dimension: 20
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (20, 10, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_built_fixed_shape_resizer_returns_expected_shape(self):
    image_resizer_text_proto = """
      fixed_shape_resizer {
        height: 10
        width: 20
      }
    """
    input_shape = (50, 25, 3)
    expected_output_shape = (10, 20, 3)
    output_shape = self._shape_of_resized_random_image_given_text_proto(
        input_shape, image_resizer_text_proto)
    self.assertEqual(output_shape, expected_output_shape)

  def test_raises_error_on_invalid_input(self):
    invalid_input = 'invalid_input'
    with self.assertRaises(ValueError):
      image_resizer_builder.build(invalid_input)


if __name__ == '__main__':
  tf.test.main()

