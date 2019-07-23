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

"""Tests for object_detection.utils.np_box_list_test."""

import numpy as np
import tensorflow as tf

from object_detection.utils import np_box_list


class BoxListTest(tf.test.TestCase):

  def test_invalid_box_data(self):
    with self.assertRaises(ValueError):
      np_box_list.BoxList([0, 0, 1, 1])

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([[0, 0, 1, 1]], dtype=int))

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([0, 1, 1, 3, 4], dtype=float))

    with self.assertRaises(ValueError):
      np_box_list.BoxList(np.array([[0, 1, 1, 3], [3, 1, 1, 5]], dtype=float))

  def test_has_field_with_existed_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertTrue(boxlist.has_field('boxes'))

  def test_has_field_with_nonexisted_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertFalse(boxlist.has_field('scores'))

  def test_get_field_with_existed_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    self.assertTrue(np.allclose(boxlist.get_field('boxes'), boxes))

  def test_get_field_with_nonexited_field(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    with self.assertRaises(ValueError):
      boxlist.get_field('scores')


class AddExtraFieldTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    self.boxlist = np_box_list.BoxList(boxes)

  def test_add_already_existed_field(self):
    with self.assertRaises(ValueError):
      self.boxlist.add_field('boxes', np.array([[0, 0, 0, 1, 0]], dtype=float))

  def test_add_invalid_field_data(self):
    with self.assertRaises(ValueError):
      self.boxlist.add_field('scores', np.array([0.5, 0.7], dtype=float))
    with self.assertRaises(ValueError):
      self.boxlist.add_field('scores',
                             np.array([0.5, 0.7, 0.9, 0.1], dtype=float))

  def test_add_single_dimensional_field_data(self):
    boxlist = self.boxlist
    scores = np.array([0.5, 0.7, 0.9], dtype=float)
    boxlist.add_field('scores', scores)
    self.assertTrue(np.allclose(scores, self.boxlist.get_field('scores')))

  def test_add_multi_dimensional_field_data(self):
    boxlist = self.boxlist
    labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                      dtype=int)
    boxlist.add_field('labels', labels)
    self.assertTrue(np.allclose(labels, self.boxlist.get_field('labels')))

  def test_get_extra_fields(self):
    boxlist = self.boxlist
    self.assertSameElements(boxlist.get_extra_fields(), [])

    scores = np.array([0.5, 0.7, 0.9], dtype=float)
    boxlist.add_field('scores', scores)
    self.assertSameElements(boxlist.get_extra_fields(), ['scores'])

    labels = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                      dtype=int)
    boxlist.add_field('labels', labels)
    self.assertSameElements(boxlist.get_extra_fields(), ['scores', 'labels'])

  def test_get_coordinates(self):
    y_min, x_min, y_max, x_max = self.boxlist.get_coordinates()

    expected_y_min = np.array([3.0, 14.0, 0.0], dtype=float)
    expected_x_min = np.array([4.0, 14.0, 0.0], dtype=float)
    expected_y_max = np.array([6.0, 15.0, 20.0], dtype=float)
    expected_x_max = np.array([8.0, 15.0, 20.0], dtype=float)

    self.assertTrue(np.allclose(y_min, expected_y_min))
    self.assertTrue(np.allclose(x_min, expected_x_min))
    self.assertTrue(np.allclose(y_max, expected_y_max))
    self.assertTrue(np.allclose(x_max, expected_x_max))

  def test_num_boxes(self):
    boxes = np.array([[0., 0., 100., 100.], [10., 30., 50., 70.]], dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    expected_num_boxes = 2
    self.assertEquals(boxlist.num_boxes(), expected_num_boxes)


if __name__ == '__main__':
  tf.test.main()
