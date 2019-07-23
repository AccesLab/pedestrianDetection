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

"""Tests for object_detection.core.bipartite_matcher."""

import tensorflow as tf

from object_detection.matchers import bipartite_matcher


class GreedyBipartiteMatcherTest(tf.test.TestCase):

  def test_get_expected_matches_when_all_rows_are_valid(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    num_valid_rows = 2
    expected_match_results = [-1, 1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, num_valid_rows=num_valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_valid_rows_set_to_minus_one(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    num_valid_rows = -1
    expected_match_results = [-1, 1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, num_valid_rows=num_valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_no_matches_with_zero_valid_rows(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    num_valid_rows = 0
    expected_match_results = [-1, -1, -1]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, num_valid_rows=num_valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)

  def test_get_expected_matches_with_only_one_valid_row(self):
    similarity_matrix = tf.constant([[0.50, 0.1, 0.8], [0.15, 0.2, 0.3]])
    num_valid_rows = 1
    expected_match_results = [-1, -1, 0]

    matcher = bipartite_matcher.GreedyBipartiteMatcher()
    match = matcher.match(similarity_matrix, num_valid_rows=num_valid_rows)
    with self.test_session() as sess:
      match_results_out = sess.run(match._match_results)
      self.assertAllEqual(match_results_out, expected_match_results)


if __name__ == '__main__':
  tf.test.main()
