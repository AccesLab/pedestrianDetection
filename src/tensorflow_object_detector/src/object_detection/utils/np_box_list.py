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

"""Numpy BoxList classes and functions."""

import numpy as np
from six import moves


class BoxList(object):
  """Box collection.

  BoxList represents a list of bounding boxes as numpy array, where each
  bounding box is represented as a row of 4 numbers,
  [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
  given list correspond to a single image.

  Optionally, users can add additional related fields (such as
  objectness/classification scores).
  """

  def __init__(self, data):
    """Constructs box collection.

    Args:
      data: a numpy array of shape [N, 4] representing box coordinates

    Raises:
      ValueError: if bbox data is not a numpy array
      ValueError: if invalid dimensions for bbox data
    """
    if not isinstance(data, np.ndarray):
      raise ValueError('data must be a numpy array.')
    if len(data.shape) != 2 or data.shape[1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    if data.dtype != np.float32 and data.dtype != np.float64:
      raise ValueError('Invalid data type for box data: float is required.')
    if not self._is_valid_boxes(data):
      raise ValueError('Invalid box data. data must be a numpy array of '
                       'N*[y_min, x_min, y_max, x_max]')
    self.data = {'boxes': data}

  def num_boxes(self):
    """Return number of boxes held in collections."""
    return self.data['boxes'].shape[0]

  def get_extra_fields(self):
    """Return all non-box fields."""
    return [k for k in self.data.keys() if k != 'boxes']

  def has_field(self, field):
    return field in self.data

  def add_field(self, field, field_data):
    """Add data to a specified field.

    Args:
      field: a string parameter used to speficy a related field to be accessed.
      field_data: a numpy array of [N, ...] representing the data associated
          with the field.
    Raises:
      ValueError: if the field is already exist or the dimension of the field
          data does not matches the number of boxes.
    """
    if self.has_field(field):
      raise ValueError('Field ' + field + 'already exists')
    if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
      raise ValueError('Invalid dimensions for field data')
    self.data[field] = field_data

  def get(self):
    """Convenience function for accesssing box coordinates.

    Returns:
      a numpy array of shape [N, 4] representing box corners
    """
    return self.get_field('boxes')

  def get_field(self, field):
    """Accesses data associated with the specified field in the box collection.

    Args:
      field: a string parameter used to speficy a related field to be accessed.

    Returns:
      a numpy 1-d array representing data of an associated field

    Raises:
      ValueError: if invalid field
    """
    if not self.has_field(field):
      raise ValueError('field {} does not exist'.format(field))
    return self.data[field]

  def get_coordinates(self):
    """Get corner coordinates of boxes.

    Returns:
     a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
    """
    box_coordinates = self.get()
    y_min = box_coordinates[:, 0]
    x_min = box_coordinates[:, 1]
    y_max = box_coordinates[:, 2]
    x_max = box_coordinates[:, 3]
    return [y_min, x_min, y_max, x_max]

  def _is_valid_boxes(self, data):
    """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].

    Args:
      data: a numpy array of shape [N, 4] representing box coordinates

    Returns:
      a boolean indicating whether all ymax of boxes are equal or greater than
          ymin, and all xmax of boxes are equal or greater than xmin.
    """
    if data.shape[0] > 0:
      for i in moves.range(data.shape[0]):
        if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
          return False
    return True
