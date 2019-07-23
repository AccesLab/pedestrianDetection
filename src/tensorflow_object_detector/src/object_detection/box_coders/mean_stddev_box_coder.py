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

"""Mean stddev box coder.

This box coder use the following coding schema to encode boxes:
rel_code = (box_corner - anchor_corner_mean) / anchor_corner_stddev.
"""
from object_detection.core import box_coder
from object_detection.core import box_list


class MeanStddevBoxCoder(box_coder.BoxCoder):
  """Mean stddev box coder."""

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of N anchors.  We assume that anchors has an associated
        stddev field.

    Returns:
      a tensor representing N anchor-encoded boxes
    Raises:
      ValueError: if the anchors BoxList does not have a stddev field
    """
    if not anchors.has_field('stddev'):
      raise ValueError('anchors must have a stddev field')
    box_corners = boxes.get()
    means = anchors.get()
    stddev = anchors.get_field('stddev')
    return (box_corners - means) / stddev

  def _decode(self, rel_codes, anchors):
    """Decode.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.  We assume that anchors has an associated
        stddev field.

    Returns:
      boxes: BoxList holding N bounding boxes
    Raises:
      ValueError: if the anchors BoxList does not have a stddev field
    """
    if not anchors.has_field('stddev'):
      raise ValueError('anchors must have a stddev field')
    means = anchors.get()
    stddevs = anchors.get_field('stddev')
    box_corners = rel_codes * stddevs + means
    return box_list.BoxList(box_corners)
