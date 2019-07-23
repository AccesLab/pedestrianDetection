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

"""A function to build an object detection anchor generator from config."""

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiple_grid_anchor_generator
from object_detection.protos import anchor_generator_pb2


def build(anchor_generator_config):
  """Builds an anchor generator based on the config.

  Args:
    anchor_generator_config: An anchor_generator.proto object containing the
      config for the desired anchor generator.

  Returns:
    Anchor generator based on the config.

  Raises:
    ValueError: On empty anchor generator proto.
  """
  if not isinstance(anchor_generator_config,
                    anchor_generator_pb2.AnchorGenerator):
    raise ValueError('anchor_generator_config not of type '
                     'anchor_generator_pb2.AnchorGenerator')
  if anchor_generator_config.WhichOneof(
      'anchor_generator_oneof') == 'grid_anchor_generator':
    grid_anchor_generator_config = anchor_generator_config.grid_anchor_generator
    return grid_anchor_generator.GridAnchorGenerator(
        scales=[float(scale) for scale in grid_anchor_generator_config.scales],
        aspect_ratios=[float(aspect_ratio)
                       for aspect_ratio
                       in grid_anchor_generator_config.aspect_ratios],
        base_anchor_size=[grid_anchor_generator_config.height,
                          grid_anchor_generator_config.width],
        anchor_stride=[grid_anchor_generator_config.height_stride,
                       grid_anchor_generator_config.width_stride],
        anchor_offset=[grid_anchor_generator_config.height_offset,
                       grid_anchor_generator_config.width_offset])
  elif anchor_generator_config.WhichOneof(
      'anchor_generator_oneof') == 'ssd_anchor_generator':
    ssd_anchor_generator_config = anchor_generator_config.ssd_anchor_generator
    return multiple_grid_anchor_generator.create_ssd_anchors(
        num_layers=ssd_anchor_generator_config.num_layers,
        min_scale=ssd_anchor_generator_config.min_scale,
        max_scale=ssd_anchor_generator_config.max_scale,
        aspect_ratios=ssd_anchor_generator_config.aspect_ratios,
        reduce_boxes_in_lowest_layer=(ssd_anchor_generator_config
                                      .reduce_boxes_in_lowest_layer))
  else:
    raise ValueError('Empty anchor generator.')

