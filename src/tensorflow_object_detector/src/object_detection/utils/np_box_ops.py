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

"""Operations for [N, 4] numpy arrays representing bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  intersect = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
  return intersect / union


def ioa(boxes1, boxes2):
  """Computes pairwise intersection-over-area between box collections.

  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
  intersect = intersection(boxes1, boxes2)
  areas = np.expand_dims(area(boxes2), axis=0)
  return intersect / areas
