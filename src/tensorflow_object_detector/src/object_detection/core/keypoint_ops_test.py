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

"""Tests for object_detection.core.keypoint_ops."""
import numpy as np
import tensorflow as tf

from object_detection.core import keypoint_ops


class KeypointOpsTest(tf.test.TestCase):
  """Tests for common keypoint operations."""

  def test_scale(self):
    keypoints = tf.constant([
        [[0.0, 0.0], [100.0, 200.0]],
        [[50.0, 120.0], [100.0, 140.0]]
    ])
    y_scale = tf.constant(1.0 / 100)
    x_scale = tf.constant(1.0 / 200)

    expected_keypoints = tf.constant([
        [[0., 0.], [1.0, 1.0]],
        [[0.5, 0.6], [1.0, 0.7]]
    ])
    output = keypoint_ops.scale(keypoints, y_scale, x_scale)

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_clip_to_window(self):
    keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])
    window = tf.constant([0.25, 0.25, 0.75, 0.75])

    expected_keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.25], [0.75, 0.75]]
    ])
    output = keypoint_ops.clip_to_window(keypoints, window)

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_prune_outside_window(self):
    keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])
    window = tf.constant([0.25, 0.25, 0.75, 0.75])

    expected_keypoints = tf.constant([[[0.25, 0.5], [0.75, 0.75]],
                                      [[np.nan, np.nan], [np.nan, np.nan]]])
    output = keypoint_ops.prune_outside_window(keypoints, window)

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_change_coordinate_frame(self):
    keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])
    window = tf.constant([0.25, 0.25, 0.75, 0.75])

    expected_keypoints = tf.constant([
        [[0, 0.5], [1.0, 1.0]],
        [[0.5, -0.5], [1.5, 1.5]]
    ])
    output = keypoint_ops.change_coordinate_frame(keypoints, window)

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_to_normalized_coordinates(self):
    keypoints = tf.constant([
        [[10., 30.], [30., 45.]],
        [[20., 0.], [40., 60.]]
    ])
    output = keypoint_ops.to_normalized_coordinates(
        keypoints, 40, 60)
    expected_keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_to_normalized_coordinates_already_normalized(self):
    keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])
    output = keypoint_ops.to_normalized_coordinates(
        keypoints, 40, 60)

    with self.test_session() as sess:
      with self.assertRaisesOpError('assertion failed'):
        sess.run(output)

  def test_to_absolute_coordinates(self):
    keypoints = tf.constant([
        [[0.25, 0.5], [0.75, 0.75]],
        [[0.5, 0.0], [1.0, 1.0]]
    ])
    output = keypoint_ops.to_absolute_coordinates(
        keypoints, 40, 60)
    expected_keypoints = tf.constant([
        [[10., 30.], [30., 45.]],
        [[20., 0.], [40., 60.]]
    ])

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)

  def test_to_absolute_coordinates_already_absolute(self):
    keypoints = tf.constant([
        [[10., 30.], [30., 45.]],
        [[20., 0.], [40., 60.]]
    ])
    output = keypoint_ops.to_absolute_coordinates(
        keypoints, 40, 60)

    with self.test_session() as sess:
      with self.assertRaisesOpError('assertion failed'):
        sess.run(output)

  def test_flip_horizontal(self):
    keypoints = tf.constant([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]
    ])
    flip_permutation = [0, 2, 1]

    expected_keypoints = tf.constant([
        [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]],
        [[0.4, 0.6], [0.6, 0.4], [0.5, 0.5]],
    ])
    output = keypoint_ops.flip_horizontal(keypoints, 0.5, flip_permutation)

    with self.test_session() as sess:
      output_, expected_keypoints_ = sess.run([output, expected_keypoints])
      self.assertAllClose(output_, expected_keypoints_)


if __name__ == '__main__':
  tf.test.main()
