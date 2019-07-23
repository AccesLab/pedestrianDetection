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

"""Tests for object_detection.models.ssd_inception_v2_feature_extractor."""
import numpy as np
import tensorflow as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_inception_v2_feature_extractor


class SsdInceptionV2FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase,
    tf.test.TestCase):

  def _create_feature_extractor(self, depth_multiplier):
    """Constructs a SsdInceptionV2FeatureExtractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
    Returns:
      an ssd_inception_v2_feature_extractor.SsdInceptionV2FeatureExtractor.
    """
    min_depth = 32
    conv_hyperparams = {}
    return ssd_inception_v2_feature_extractor.SSDInceptionV2FeatureExtractor(
        depth_multiplier, min_depth, conv_hyperparams)

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = [(4, 8, 8, 576), (4, 4, 4, 1024),
                                  (4, 2, 2, 512), (4, 1, 1, 256),
                                  (4, 1, 1, 256), (4, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    expected_feature_map_shape = [(4, 19, 19, 576), (4, 10, 10, 1024),
                                  (4, 5, 5, 512), (4, 3, 3, 256),
                                  (4, 2, 2, 256), (4, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 0.5**12
    expected_feature_map_shape = [(4, 19, 19, 128), (4, 10, 10, 128),
                                  (4, 5, 5, 32), (4, 3, 3, 32),
                                  (4, 2, 2, 32), (4, 1, 1, 32)]
    self.check_extract_features_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_extract_features_raises_error_with_invalid_image_size(self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    scope_name = 'InceptionV2'
    self.check_feature_extractor_variables_under_scope(depth_multiplier,
                                                       scope_name)


if __name__ == '__main__':
  tf.test.main()
