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

"""Tests for object_detection.box_coder.mean_stddev_boxcoder."""

import tensorflow as tf

from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list


class MeanStddevBoxCoderTest(tf.test.TestCase):

  def testGetCorrectRelativeCodesAfterEncoding(self):
    box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    expected_rel_codes = [[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]]
    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
    prior_stddevs = tf.constant(2 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    rel_codes = coder.encode(boxes, priors)
    with self.test_session() as sess:
      rel_codes_out = sess.run(rel_codes)
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def testGetCorrectBoxesAfterDecoding(self):
    rel_codes = tf.constant([[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]])
    expected_box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
    prior_stddevs = tf.constant(2 * [4 * [.1]])
    priors = box_list.BoxList(prior_means)
    priors.add_field('stddev', prior_stddevs)

    coder = mean_stddev_box_coder.MeanStddevBoxCoder()
    decoded_boxes = coder.decode(rel_codes, priors)
    decoded_box_corners = decoded_boxes.get()
    with self.test_session() as sess:
      decoded_out = sess.run(decoded_box_corners)
      self.assertAllClose(decoded_out, expected_box_corners)


if __name__ == '__main__':
  tf.test.main()
