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

"""Contains functions which are convenient for unit testing."""
import numpy as np
import tensorflow as tf

from object_detection.core import anchor_generator
from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import box_predictor
from object_detection.core import matcher


class MockBoxCoder(box_coder.BoxCoder):
  """Simple `difference` BoxCoder."""

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    return boxes.get() - anchors.get()

  def _decode(self, rel_codes, anchors):
    return box_list.BoxList(rel_codes + anchors.get())


class MockBoxPredictor(box_predictor.BoxPredictor):
  """Simple box predictor that ignores inputs and outputs all zeros."""

  def __init__(self, is_training, num_classes):
    super(MockBoxPredictor, self).__init__(is_training, num_classes)

  def _predict(self, image_features, num_predictions_per_location):
    batch_size = image_features.get_shape().as_list()[0]
    num_anchors = (image_features.get_shape().as_list()[1]
                   * image_features.get_shape().as_list()[2])
    code_size = 4
    zero = tf.reduce_sum(0 * image_features)
    box_encodings = zero + tf.zeros(
        (batch_size, num_anchors, 1, code_size), dtype=tf.float32)
    class_predictions_with_background = zero + tf.zeros(
        (batch_size, num_anchors, self.num_classes + 1), dtype=tf.float32)
    return {box_predictor.BOX_ENCODINGS: box_encodings,
            box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background}


class MockAnchorGenerator(anchor_generator.AnchorGenerator):
  """Mock anchor generator."""

  def name_scope(self):
    return 'MockAnchorGenerator'

  def num_anchors_per_location(self):
    return [1]

  def _generate(self, feature_map_shape_list):
    num_anchors = sum([shape[0] * shape[1] for shape in feature_map_shape_list])
    return box_list.BoxList(tf.zeros((num_anchors, 4), dtype=tf.float32))


class MockMatcher(matcher.Matcher):
  """Simple matcher that matches first anchor to first groundtruth box."""

  def _match(self, similarity_matrix):
    return tf.constant([0, -1, -1, -1], dtype=tf.int32)


def create_diagonal_gradient_image(height, width, depth):
  """Creates pyramid image. Useful for testing.

  For example, pyramid_image(5, 6, 1) looks like:
  # [[[ 5.  4.  3.  2.  1.  0.]
  #   [ 6.  5.  4.  3.  2.  1.]
  #   [ 7.  6.  5.  4.  3.  2.]
  #   [ 8.  7.  6.  5.  4.  3.]
  #   [ 9.  8.  7.  6.  5.  4.]]]

  Args:
    height: height of image
    width: width of image
    depth: depth of image

  Returns:
    pyramid image
  """
  row = np.arange(height)
  col = np.arange(width)[::-1]
  image_layer = np.expand_dims(row, 1) + col
  image_layer = np.expand_dims(image_layer, 2)

  image = image_layer
  for i in range(1, depth):
    image = np.concatenate((image, image_layer * pow(10, i)), 2)

  return image.astype(np.float32)


def create_random_boxes(num_boxes, max_height, max_width):
  """Creates random bounding boxes of specific maximum height and width.

  Args:
    num_boxes: number of boxes.
    max_height: maximum height of boxes.
    max_width: maximum width of boxes.

  Returns:
    boxes: numpy array of shape [num_boxes, 4]. Each row is in form
        [y_min, x_min, y_max, x_max].
  """

  y_1 = np.random.uniform(size=(1, num_boxes)) * max_height
  y_2 = np.random.uniform(size=(1, num_boxes)) * max_height
  x_1 = np.random.uniform(size=(1, num_boxes)) * max_width
  x_2 = np.random.uniform(size=(1, num_boxes)) * max_width

  boxes = np.zeros(shape=(num_boxes, 4))
  boxes[:, 0] = np.minimum(y_1, y_2)
  boxes[:, 1] = np.minimum(x_1, x_2)
  boxes[:, 2] = np.maximum(y_1, y_2)
  boxes[:, 3] = np.maximum(x_1, x_2)

  return boxes.astype(np.float32)
