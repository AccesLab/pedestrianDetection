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

"""Builder for region similarity calculators."""

from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2


def build(region_similarity_calculator_config):
  """Builds region similarity calculator based on the configuration.

  Builds one of [IouSimilarity, IoaSimilarity, NegSqDistSimilarity] objects. See
  core/region_similarity_calculator.proto for details.

  Args:
    region_similarity_calculator_config: RegionSimilarityCalculator
      configuration proto.

  Returns:
    region_similarity_calculator: RegionSimilarityCalculator object.

  Raises:
    ValueError: On unknown region similarity calculator.
  """

  if not isinstance(
      region_similarity_calculator_config,
      region_similarity_calculator_pb2.RegionSimilarityCalculator):
    raise ValueError(
        'region_similarity_calculator_config not of type '
        'region_similarity_calculator_pb2.RegionsSimilarityCalculator')

  similarity_calculator = region_similarity_calculator_config.WhichOneof(
      'region_similarity')
  if similarity_calculator == 'iou_similarity':
    return region_similarity_calculator.IouSimilarity()
  if similarity_calculator == 'ioa_similarity':
    return region_similarity_calculator.IoaSimilarity()
  if similarity_calculator == 'neg_sq_dist_similarity':
    return region_similarity_calculator.NegSqDistSimilarity()

  raise ValueError('Unknown region similarity calculator.')

