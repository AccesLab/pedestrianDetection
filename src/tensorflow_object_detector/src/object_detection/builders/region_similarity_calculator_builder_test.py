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

"""Tests for region_similarity_calculator_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import region_similarity_calculator_builder
from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2 as sim_calc_pb2


class RegionSimilarityCalculatorBuilderTest(tf.test.TestCase):

  def testBuildIoaSimilarityCalculator(self):
    similarity_calc_text_proto = """
      ioa_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IoaSimilarity))

  def testBuildIouSimilarityCalculator(self):
    similarity_calc_text_proto = """
      iou_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IouSimilarity))

  def testBuildNegSqDistSimilarityCalculator(self):
    similarity_calc_text_proto = """
      neg_sq_dist_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.
                               NegSqDistSimilarity))


if __name__ == '__main__':
  tf.test.main()
