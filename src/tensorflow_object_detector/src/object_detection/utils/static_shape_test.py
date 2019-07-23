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

"""Tests for object_detection.utils.static_shape."""

import tensorflow as tf

from object_detection.utils import static_shape


class StaticShapeTest(tf.test.TestCase):

  def test_return_correct_batchSize(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(32, static_shape.get_batch_size(tensor_shape))

  def test_return_correct_height(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(299, static_shape.get_height(tensor_shape))

  def test_return_correct_width(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(384, static_shape.get_width(tensor_shape))

  def test_return_correct_depth(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(3, static_shape.get_depth(tensor_shape))

  def test_die_on_tensor_shape_with_rank_three(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384])
    with self.assertRaises(ValueError):
      static_shape.get_batch_size(tensor_shape)
      static_shape.get_height(tensor_shape)
      static_shape.get_width(tensor_shape)
      static_shape.get_depth(tensor_shape)

if __name__ == '__main__':
  tf.test.main()
