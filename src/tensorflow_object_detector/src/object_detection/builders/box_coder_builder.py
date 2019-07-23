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

"""A function to build an object detection box coder from configuration."""
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.protos import box_coder_pb2


def build(box_coder_config):
  """Builds a box coder object based on the box coder config.

  Args:
    box_coder_config: A box_coder.proto object containing the config for the
      desired box coder.

  Returns:
    BoxCoder based on the config.

  Raises:
    ValueError: On empty box coder proto.
  """
  if not isinstance(box_coder_config, box_coder_pb2.BoxCoder):
    raise ValueError('box_coder_config not of type box_coder_pb2.BoxCoder.')

  if box_coder_config.WhichOneof('box_coder_oneof') == 'faster_rcnn_box_coder':
    return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[
        box_coder_config.faster_rcnn_box_coder.y_scale,
        box_coder_config.faster_rcnn_box_coder.x_scale,
        box_coder_config.faster_rcnn_box_coder.height_scale,
        box_coder_config.faster_rcnn_box_coder.width_scale
    ])
  if (box_coder_config.WhichOneof('box_coder_oneof') ==
      'mean_stddev_box_coder'):
    return mean_stddev_box_coder.MeanStddevBoxCoder()
  if box_coder_config.WhichOneof('box_coder_oneof') == 'square_box_coder':
    return square_box_coder.SquareBoxCoder(scale_factors=[
        box_coder_config.square_box_coder.y_scale,
        box_coder_config.square_box_coder.x_scale,
        box_coder_config.square_box_coder.length_scale
    ])
  raise ValueError('Empty box coder.')
