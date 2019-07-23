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

"""Tests for matcher_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import matcher_builder
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.protos import matcher_pb2


class MatcherBuilderTest(tf.test.TestCase):

  def test_build_arg_max_matcher_with_defaults(self):
    matcher_text_proto = """
      argmax_matcher {
      }
    """
    matcher_proto = matcher_pb2.Matcher()
    text_format.Merge(matcher_text_proto, matcher_proto)
    matcher_object = matcher_builder.build(matcher_proto)
    self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
    self.assertAlmostEqual(matcher_object._matched_threshold, 0.5)
    self.assertAlmostEqual(matcher_object._unmatched_threshold, 0.5)
    self.assertTrue(matcher_object._negatives_lower_than_unmatched)
    self.assertFalse(matcher_object._force_match_for_each_row)

  def test_build_arg_max_matcher_without_thresholds(self):
    matcher_text_proto = """
      argmax_matcher {
        ignore_thresholds: true
      }
    """
    matcher_proto = matcher_pb2.Matcher()
    text_format.Merge(matcher_text_proto, matcher_proto)
    matcher_object = matcher_builder.build(matcher_proto)
    self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
    self.assertEqual(matcher_object._matched_threshold, None)
    self.assertEqual(matcher_object._unmatched_threshold, None)
    self.assertTrue(matcher_object._negatives_lower_than_unmatched)
    self.assertFalse(matcher_object._force_match_for_each_row)

  def test_build_arg_max_matcher_with_non_default_parameters(self):
    matcher_text_proto = """
      argmax_matcher {
        matched_threshold: 0.7
        unmatched_threshold: 0.3
        negatives_lower_than_unmatched: false
        force_match_for_each_row: true
      }
    """
    matcher_proto = matcher_pb2.Matcher()
    text_format.Merge(matcher_text_proto, matcher_proto)
    matcher_object = matcher_builder.build(matcher_proto)
    self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
    self.assertAlmostEqual(matcher_object._matched_threshold, 0.7)
    self.assertAlmostEqual(matcher_object._unmatched_threshold, 0.3)
    self.assertFalse(matcher_object._negatives_lower_than_unmatched)
    self.assertTrue(matcher_object._force_match_for_each_row)

  def test_build_bipartite_matcher(self):
    matcher_text_proto = """
      bipartite_matcher {
      }
    """
    matcher_proto = matcher_pb2.Matcher()
    text_format.Merge(matcher_text_proto, matcher_proto)
    matcher_object = matcher_builder.build(matcher_proto)
    self.assertTrue(
        isinstance(matcher_object, bipartite_matcher.GreedyBipartiteMatcher))

  def test_raise_error_on_empty_matcher(self):
    matcher_text_proto = """
    """
    matcher_proto = matcher_pb2.Matcher()
    text_format.Merge(matcher_text_proto, matcher_proto)
    with self.assertRaises(ValueError):
      matcher_builder.build(matcher_proto)


if __name__ == '__main__':
  tf.test.main()
