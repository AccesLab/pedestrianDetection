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

"""SSDFeatureExtractor for InceptionV2 features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from nets import inception_v2

slim = tf.contrib.slim


class SSDInceptionV2FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using InceptionV2 features."""

  def __init__(self,
               depth_multiplier,
               min_depth,
               conv_hyperparams,
               reuse_weights=None):
    """InceptionV2 Feature Extractor for SSD Models.

    Args:
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(SSDInceptionV2FeatureExtractor, self).__init__(
        depth_multiplier, min_depth, conv_hyperparams, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('InceptionV2',
                               reuse=self._reuse_weights) as scope:
          _, image_features = inception_v2.inception_v2_base(
              preprocessed_inputs,
              final_endpoint='Mixed_5c',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()
