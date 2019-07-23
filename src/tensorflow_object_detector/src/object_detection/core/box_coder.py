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

"""Base box coder.

Box coders convert between coordinate frames, namely image-centric
(with (0,0) on the top left of image) and anchor-centric (with (0,0) being
defined by a specific anchor).

Users of a BoxCoder can call two methods:
 encode: which encodes a box with respect to a given anchor
  (or rather, a tensor of boxes wrt a corresponding tensor of anchors) and
 decode: which inverts this encoding with a decode operation.
In both cases, the arguments are assumed to be in 1-1 correspondence already;
it is not the job of a BoxCoder to perform matching.
"""
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

import tensorflow as tf


# Box coder types.
FASTER_RCNN = 'faster_rcnn'
KEYPOINT = 'keypoint'
MEAN_STDDEV = 'mean_stddev'
SQUARE = 'square'


class BoxCoder(object):
  """Abstract base class for box coder."""
  __metaclass__ = ABCMeta

  @abstractproperty
  def code_size(self):
    """Return the size of each code.

    This number is a constant and should agree with the output of the `encode`
    op (e.g. if rel_codes is the output of self.encode(...), then it should have
    shape [N, code_size()]).  This abstractproperty should be overridden by
    implementations.

    Returns:
      an integer constant
    """
    pass

  def encode(self, boxes, anchors):
    """Encode a box list relative to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded
      anchors: BoxList of N anchors

    Returns:
      a tensor representing N relative-encoded boxes
    """
    with tf.name_scope('Encode'):
      return self._encode(boxes, anchors)

  def decode(self, rel_codes, anchors):
    """Decode boxes that are encoded relative to an anchor collection.

    Args:
      rel_codes: a tensor representing N relative-encoded boxes
      anchors: BoxList of anchors

    Returns:
      boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
        with corners y_min, x_min, y_max, x_max)
    """
    with tf.name_scope('Decode'):
      return self._decode(rel_codes, anchors)

  @abstractmethod
  def _encode(self, boxes, anchors):
    """Method to be overriden by implementations.

    Args:
      boxes: BoxList holding N boxes to be encoded
      anchors: BoxList of N anchors

    Returns:
      a tensor representing N relative-encoded boxes
    """
    pass

  @abstractmethod
  def _decode(self, rel_codes, anchors):
    """Method to be overriden by implementations.

    Args:
      rel_codes: a tensor representing N relative-encoded boxes
      anchors: BoxList of anchors

    Returns:
      boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
        with corners y_min, x_min, y_max, x_max)
    """
    pass


def batch_decode(encoded_boxes, box_coder, anchors):
  """Decode a batch of encoded boxes.

  This op takes a batch of encoded bounding boxes and transforms
  them to a batch of bounding boxes specified by their corners in
  the order of [y_min, x_min, y_max, x_max].

  Args:
    encoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      code_size] representing the location of the objects.
    box_coder: a BoxCoder object.
    anchors: a BoxList of anchors used to encode `encoded_boxes`.

  Returns:
    decoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      coder_size] representing the corners of the objects in the order
      of [y_min, x_min, y_max, x_max].

  Raises:
    ValueError: if batch sizes of the inputs are inconsistent, or if
    the number of anchors inferred from encoded_boxes and anchors are
    inconsistent.
  """
  encoded_boxes.get_shape().assert_has_rank(3)
  if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
    raise ValueError('The number of anchors inferred from encoded_boxes'
                     ' and anchors are inconsistent: shape[1] of encoded_boxes'
                     ' %s should be equal to the number of anchors: %s.' %
                     (encoded_boxes.get_shape()[1].value,
                      anchors.num_boxes_static()))

  decoded_boxes = tf.stack([
      box_coder.decode(boxes, anchors).get()
      for boxes in tf.unstack(encoded_boxes)
  ])
  return decoded_boxes
