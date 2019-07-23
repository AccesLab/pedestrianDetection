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
"""Tests for object_detection.grid_anchor_generator."""

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator


class GridAnchorGeneratorTest(tf.test.TestCase):

  def test_construct_single_anchor(self):
    """Builds a 1x1 anchor grid to test the size of the output boxes."""
    scales = [0.5, 1.0, 2.0]
    aspect_ratios = [0.25, 1.0, 4.0]
    anchor_offset = [7, -3]
    exp_anchor_corners = [[-121, -35, 135, 29], [-249, -67, 263, 61],
                          [-505, -131, 519, 125], [-57, -67, 71, 61],
                          [-121, -131, 135, 125], [-249, -259, 263, 253],
                          [-25, -131, 39, 125], [-57, -259, 71, 253],
                          [-121, -515, 135, 509]]

    anchor_generator = grid_anchor_generator.GridAnchorGenerator(
        scales, aspect_ratios,
        anchor_offset=anchor_offset)
    anchors = anchor_generator.generate(feature_map_shape_list=[(1, 1)])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid(self):
    base_anchor_size = [10, 10]
    anchor_stride = [19, 19]
    anchor_offset = [0, 0]
    scales = [0.5, 1.0, 2.0]
    aspect_ratios = [1.0]

    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]

    anchor_generator = grid_anchor_generator.GridAnchorGenerator(
        scales,
        aspect_ratios,
        base_anchor_size=base_anchor_size,
        anchor_stride=anchor_stride,
        anchor_offset=anchor_offset)

    anchors = anchor_generator.generate(feature_map_shape_list=[(2, 2)])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)


if __name__ == '__main__':
  tf.test.main()
