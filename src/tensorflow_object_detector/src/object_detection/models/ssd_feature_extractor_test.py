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

"""Base test class SSDFeatureExtractors."""

from abc import abstractmethod

import numpy as np
import tensorflow as tf


class SsdFeatureExtractorTestBase(object):

  def _validate_features_shape(self,
                               feature_extractor,
                               preprocessed_inputs,
                               expected_feature_map_shapes):
    """Checks the extracted features are of correct shape.

    Args:
      feature_extractor: The feature extractor to test.
      preprocessed_inputs: A [batch, height, width, 3] tensor to extract
                           features with.
      expected_feature_map_shapes: The expected shape of the extracted features.
    """
    feature_maps = feature_extractor.extract_features(preprocessed_inputs)
    feature_map_shapes = [tf.shape(feature_map) for feature_map in feature_maps]
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      feature_map_shapes_out = sess.run(feature_map_shapes)
      for shape_out, exp_shape_out in zip(
          feature_map_shapes_out, expected_feature_map_shapes):
        self.assertAllEqual(shape_out, exp_shape_out)

  @abstractmethod
  def _create_feature_extractor(self, depth_multiplier):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    pass

  def check_extract_features_returns_correct_shape(
      self,
      image_height,
      image_width,
      depth_multiplier,
      expected_feature_map_shapes_out):
    feature_extractor = self._create_feature_extractor(depth_multiplier)
    preprocessed_inputs = tf.random_uniform(
        [4, image_height, image_width, 3], dtype=tf.float32)
    self._validate_features_shape(
        feature_extractor, preprocessed_inputs, expected_feature_map_shapes_out)

  def check_extract_features_raises_error_with_invalid_image_size(
      self,
      image_height,
      image_width,
      depth_multiplier):
    feature_extractor = self._create_feature_extractor(depth_multiplier)
    preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
    feature_maps = feature_extractor.extract_features(preprocessed_inputs)
    test_preprocessed_image = np.random.rand(4, image_height, image_width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(feature_maps,
                 feed_dict={preprocessed_inputs: test_preprocessed_image})

  def check_feature_extractor_variables_under_scope(self,
                                                    depth_multiplier,
                                                    scope_name):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = self._create_feature_extractor(depth_multiplier)
      preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
      feature_extractor.extract_features(preprocessed_inputs)
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      for variable in variables:
        self.assertTrue(variable.name.startswith(scope_name))
