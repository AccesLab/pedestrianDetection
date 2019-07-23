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

"""Tests for object_detection.utils.test_utils."""

import numpy as np
import tensorflow as tf

from object_detection.utils import test_utils


class TestUtilsTest(tf.test.TestCase):

  def test_diagonal_gradient_image(self):
    """Tests if a good pyramid image is created."""
    pyramid_image = test_utils.create_diagonal_gradient_image(3, 4, 2)

    # Test which is easy to understand.
    expected_first_channel = np.array([[3, 2, 1, 0],
                                       [4, 3, 2, 1],
                                       [5, 4, 3, 2]], dtype=np.float32)
    self.assertAllEqual(np.squeeze(pyramid_image[:, :, 0]),
                        expected_first_channel)

    # Actual test.
    expected_image = np.array([[[3, 30],
                                [2, 20],
                                [1, 10],
                                [0, 0]],
                               [[4, 40],
                                [3, 30],
                                [2, 20],
                                [1, 10]],
                               [[5, 50],
                                [4, 40],
                                [3, 30],
                                [2, 20]]], dtype=np.float32)

    self.assertAllEqual(pyramid_image, expected_image)

  def test_random_boxes(self):
    """Tests if valid random boxes are created."""
    num_boxes = 1000
    max_height = 3
    max_width = 5
    boxes = test_utils.create_random_boxes(num_boxes,
                                           max_height,
                                           max_width)

    true_column = np.ones(shape=(num_boxes)) == 1
    self.assertAllEqual(boxes[:, 0] < boxes[:, 2], true_column)
    self.assertAllEqual(boxes[:, 1] < boxes[:, 3], true_column)

    self.assertTrue(boxes[:, 0].min() >= 0)
    self.assertTrue(boxes[:, 1].min() >= 0)
    self.assertTrue(boxes[:, 2].max() <= max_height)
    self.assertTrue(boxes[:, 3].max() <= max_width)


if __name__ == '__main__':
  tf.test.main()
