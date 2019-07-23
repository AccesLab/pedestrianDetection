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

"""Bipartite matcher implementation."""

import tensorflow as tf

from tensorflow.contrib.image.python.ops import image_ops
from object_detection.core import matcher


class GreedyBipartiteMatcher(matcher.Matcher):
  """Wraps a Tensorflow greedy bipartite matcher."""

  def _match(self, similarity_matrix, num_valid_rows=-1):
    """Bipartite matches a collection rows and columns. A greedy bi-partite.

    TODO: Add num_valid_columns options to match only that many columns with
        all the rows.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      num_valid_rows: A scalar or a 1-D tensor with one element describing the
        number of valid rows of similarity_matrix to consider for the bipartite
        matching. If set to be negative, then all rows from similarity_matrix
        are used.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    # Convert similarity matrix to distance matrix as tf.image.bipartite tries
    # to find minimum distance matches.
    distance_matrix = -1 * similarity_matrix
    _, match_results = image_ops.bipartite_match(
        distance_matrix, num_valid_rows)
    match_results = tf.reshape(match_results, [-1])
    match_results = tf.cast(match_results, tf.int32)
    return match_results
