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

"""A function to build an object detection matcher from configuration."""

from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.protos import matcher_pb2


def build(matcher_config):
  """Builds a matcher object based on the matcher config.

  Args:
    matcher_config: A matcher.proto object containing the config for the desired
      Matcher.

  Returns:
    Matcher based on the config.

  Raises:
    ValueError: On empty matcher proto.
  """
  if not isinstance(matcher_config, matcher_pb2.Matcher):
    raise ValueError('matcher_config not of type matcher_pb2.Matcher.')
  if matcher_config.WhichOneof('matcher_oneof') == 'argmax_matcher':
    matcher = matcher_config.argmax_matcher
    matched_threshold = unmatched_threshold = None
    if not matcher.ignore_thresholds:
      matched_threshold = matcher.matched_threshold
      unmatched_threshold = matcher.unmatched_threshold
    return argmax_matcher.ArgMaxMatcher(
        matched_threshold=matched_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=matcher.negatives_lower_than_unmatched,
        force_match_for_each_row=matcher.force_match_for_each_row)
  if matcher_config.WhichOneof('matcher_oneof') == 'bipartite_matcher':
    return bipartite_matcher.GreedyBipartiteMatcher()
  raise ValueError('Empty matcher.')
