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

"""Tests for object_detection.models.faster_rcnn_resnet_v1_feature_extractor."""

import numpy as np
import tensorflow as tf

from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1


class FasterRcnnResnetV1FeatureExtractorTest(tf.test.TestCase):

  def _build_feature_extractor(self,
                               first_stage_features_stride,
                               architecture='resnet_v1_101'):
    feature_extractor_map = {
        'resnet_v1_50':
            faster_rcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
        'resnet_v1_101':
            faster_rcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
        'resnet_v1_152':
            faster_rcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor
    }
    return feature_extractor_map[architecture](
        is_training=False,
        first_stage_features_stride=first_stage_features_stride,
        reuse_weights=None,
        weight_decay=0.0)

  def test_extract_proposal_features_returns_expected_size(self):
    for architecture in ['resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152']:
      feature_extractor = self._build_feature_extractor(
          first_stage_features_stride=16, architecture=architecture)
      preprocessed_inputs = tf.random_uniform(
          [4, 224, 224, 3], maxval=255, dtype=tf.float32)
      rpn_feature_map = feature_extractor.extract_proposal_features(
          preprocessed_inputs, scope='TestScope')
      features_shape = tf.shape(rpn_feature_map)

      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        features_shape_out = sess.run(features_shape)
        self.assertAllEqual(features_shape_out, [4, 14, 14, 1024])

  def test_extract_proposal_features_stride_eight(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=8)
    preprocessed_inputs = tf.random_uniform(
        [4, 224, 224, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='TestScope')
    features_shape = tf.shape(rpn_feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      features_shape_out = sess.run(features_shape)
      self.assertAllEqual(features_shape_out, [4, 28, 28, 1024])

  def test_extract_proposal_features_half_size_input(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    preprocessed_inputs = tf.random_uniform(
        [1, 112, 112, 3], maxval=255, dtype=tf.float32)
    rpn_feature_map = feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='TestScope')
    features_shape = tf.shape(rpn_feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      features_shape_out = sess.run(features_shape)
      self.assertAllEqual(features_shape_out, [1, 7, 7, 1024])

  def test_extract_proposal_features_dies_on_invalid_stride(self):
    with self.assertRaises(ValueError):
      self._build_feature_extractor(first_stage_features_stride=99)

  def test_extract_proposal_features_dies_on_very_small_images(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
    rpn_feature_map = feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='TestScope')
    features_shape = tf.shape(rpn_feature_map)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(
            features_shape,
            feed_dict={preprocessed_inputs: np.random.rand(4, 32, 32, 3)})

  def test_extract_proposal_features_dies_with_incorrect_rank_inputs(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    preprocessed_inputs = tf.random_uniform(
        [224, 224, 3], maxval=255, dtype=tf.float32)
    with self.assertRaises(ValueError):
      feature_extractor.extract_proposal_features(
          preprocessed_inputs, scope='TestScope')

  def test_extract_box_classifier_features_returns_expected_size(self):
    feature_extractor = self._build_feature_extractor(
        first_stage_features_stride=16)
    proposal_feature_maps = tf.random_uniform(
        [3, 7, 7, 1024], maxval=255, dtype=tf.float32)
    proposal_classifier_features = (
        feature_extractor.extract_box_classifier_features(
            proposal_feature_maps, scope='TestScope'))
    features_shape = tf.shape(proposal_classifier_features)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      features_shape_out = sess.run(features_shape)
      self.assertAllEqual(features_shape_out, [3, 7, 7, 2048])


if __name__ == '__main__':
  tf.test.main()
