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

"""Tests for google3.research.vale.object_detection.minibatch_sampler."""

import numpy as np
import tensorflow as tf

from object_detection.core import minibatch_sampler


class MinibatchSamplerTest(tf.test.TestCase):

  def test_subsample_indicator_when_more_true_elements_than_num_samples(self):
    np_indicator = [True, False, True, False, True, True, False]
    indicator = tf.constant(np_indicator)
    samples = minibatch_sampler.MinibatchSampler.subsample_indicator(
        indicator, 3)
    with self.test_session() as sess:
      samples_out = sess.run(samples)
      self.assertTrue(np.sum(samples_out), 3)
      self.assertAllEqual(samples_out,
                          np.logical_and(samples_out, np_indicator))

  def test_subsample_when_more_true_elements_than_num_samples_no_shape(self):
    np_indicator = [True, False, True, False, True, True, False]
    indicator = tf.placeholder(tf.bool)
    feed_dict = {indicator: np_indicator}

    samples = minibatch_sampler.MinibatchSampler.subsample_indicator(
        indicator, 3)
    with self.test_session() as sess:
      samples_out = sess.run(samples, feed_dict=feed_dict)
      self.assertTrue(np.sum(samples_out), 3)
      self.assertAllEqual(samples_out,
                          np.logical_and(samples_out, np_indicator))

  def test_subsample_indicator_when_less_true_elements_than_num_samples(self):
    np_indicator = [True, False, True, False, True, True, False]
    indicator = tf.constant(np_indicator)
    samples = minibatch_sampler.MinibatchSampler.subsample_indicator(
        indicator, 5)
    with self.test_session() as sess:
      samples_out = sess.run(samples)
      self.assertTrue(np.sum(samples_out), 4)
      self.assertAllEqual(samples_out,
                          np.logical_and(samples_out, np_indicator))

  def test_subsample_indicator_when_num_samples_is_zero(self):
    np_indicator = [True, False, True, False, True, True, False]
    indicator = tf.constant(np_indicator)
    samples_none = minibatch_sampler.MinibatchSampler.subsample_indicator(
        indicator, 0)
    with self.test_session() as sess:
      samples_none_out = sess.run(samples_none)
      self.assertAllEqual(
          np.zeros_like(samples_none_out, dtype=bool),
          samples_none_out)

  def test_subsample_indicator_when_indicator_all_false(self):
    indicator_empty = tf.zeros([0], dtype=tf.bool)
    samples_empty = minibatch_sampler.MinibatchSampler.subsample_indicator(
        indicator_empty, 4)
    with self.test_session() as sess:
      samples_empty_out = sess.run(samples_empty)
      self.assertEqual(0, samples_empty_out.size)


if __name__ == '__main__':
  tf.test.main()
