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

"""Tests for google3.research.vale.object_detection.losses."""
import math

import numpy as np
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import losses
from object_detection.core import matcher


class WeightedL2LocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], tf.float32)
    loss_op = losses.WeightedL2LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = (3 * 5 * 4) / 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectAnchorwiseLoss(self):
    batch_size = 3
    num_anchors = 16
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss(anchorwise_output=True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = np.ones((batch_size, num_anchors)) * 2
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)

  def testReturnsCorrectLossSum(self):
    batch_size = 3
    num_anchors = 16
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss(anchorwise_output=False)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    expected_loss = tf.nn.l2_loss(prediction_tensor - target_tensor)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      expected_loss_output = sess.run(expected_loss)
      self.assertAllClose(loss_output, expected_loss_output)

  def testReturnsCorrectNanLoss(self):
    batch_size = 3
    num_anchors = 10
    code_size = 4
    prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
    target_tensor = tf.concat([
        tf.zeros([batch_size, num_anchors, code_size / 2]),
        tf.ones([batch_size, num_anchors, code_size / 2]) * np.nan
    ], axis=2)
    weights = tf.ones([batch_size, num_anchors])
    loss_op = losses.WeightedL2LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                   ignore_nan_targets=True)

    expected_loss = (3 * 5 * 4) / 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, expected_loss)


class WeightedSmoothL1LocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    batch_size = 2
    num_anchors = 3
    code_size = 4
    prediction_tensor = tf.constant([[[2.5, 0, .4, 0],
                                      [0, 0, 0, 0],
                                      [0, 2.5, 0, .4]],
                                     [[3.5, 0, 0, 0],
                                      [0, .4, 0, .9],
                                      [0, 0, 1.5, 0]]], tf.float32)
    target_tensor = tf.zeros([batch_size, num_anchors, code_size])
    weights = tf.constant([[2, 1, 1],
                           [0, 3, 0]], tf.float32)
    loss_op = losses.WeightedSmoothL1LocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = 7.695
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class WeightedIOULocalizationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                      [0, 0, 1, 1],
                                      [0, 0, .5, .25]]])
    target_tensor = tf.constant([[[1.5, 0, 2.4, 1],
                                  [0, 0, 1, 1],
                                  [50, 50, 500.5, 100.25]]])
    weights = [[1.0, .5, 2.0]]
    loss_op = losses.WeightedIOULocalizationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = 2.0
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class WeightedSigmoidClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSigmoidClassificationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = -2 * math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSigmoidClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossWithClassIndices(self):
    prediction_tensor = tf.constant([[[-100, 100, -100, 100],
                                      [100, -100, -100, -100],
                                      [100, 0, -100, 100],
                                      [-100, -100, 100, -100]],
                                     [[-100, 0, 100, 100],
                                      [-100, 100, -100, 100],
                                      [100, 100, 100, 100],
                                      [0, 0, -1, 100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0, 0],
                                  [1, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 1]],
                                 [[0, 0, 1, 0],
                                  [0, 1, 0, 0],
                                  [1, 1, 1, 0],
                                  [1, 0, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    # Ignores the last class.
    class_indices = tf.constant([0, 1, 2], tf.int32)
    loss_op = losses.WeightedSigmoidClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                   class_indices=class_indices)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class WeightedSoftmaxClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [0, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 0],
                                      [-100, 100, -100],
                                      [-100, 100, -100],
                                      [100, -100, -100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, .5, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSoftmaxClassificationLoss()
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = - 1.5 * math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [0, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 0],
                                      [-100, 100, -100],
                                      [-100, 100, -100],
                                      [100, -100, -100]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, .5, 1],
                           [1, 1, 1, 0]], tf.float32)
    loss_op = losses.WeightedSoftmaxClassificationLoss(True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, - 0.5 * math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class BootstrappedSigmoidClassificationLossTest(tf.test.TestCase):

  def testReturnsCorrectLossSoftBootstrapping(self):
    prediction_tensor = tf.constant([[[-100, 100, 0],
                                      [100, -100, -100],
                                      [100, -100, -100],
                                      [-100, -100, 100]],
                                     [[-100, -100, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='soft')
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = -math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectLossHardBootstrapping(self):
    prediction_tensor = tf.constant([[[-100, 100, 0],
                                      [100, -100, -100],
                                      [100, -100, -100],
                                      [-100, -100, 100]],
                                     [[-100, -100, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='hard')
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)
    exp_loss = -math.log(.5)
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)

  def testReturnsCorrectAnchorWiseLoss(self):
    prediction_tensor = tf.constant([[[-100, 100, -100],
                                      [100, -100, -100],
                                      [100, 0, -100],
                                      [-100, -100, 100]],
                                     [[-100, 0, 100],
                                      [-100, 100, -100],
                                      [100, 100, 100],
                                      [0, 0, -1]]], tf.float32)
    target_tensor = tf.constant([[[0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 1, 1],
                                  [1, 0, 0]]], tf.float32)
    weights = tf.constant([[1, 1, 1, 1],
                           [1, 1, 1, 0]], tf.float32)
    alpha = tf.constant(.5, tf.float32)
    loss_op = losses.BootstrappedSigmoidClassificationLoss(
        alpha, bootstrap_type='hard', anchorwise_output=True)
    loss = loss_op(prediction_tensor, target_tensor, weights=weights)

    exp_loss = np.matrix([[0, 0, -math.log(.5), 0],
                          [-math.log(.5), 0, 0, 0]])
    with self.test_session() as sess:
      loss_output = sess.run(loss)
      self.assertAllClose(loss_output, exp_loss)


class HardExampleMinerTest(tf.test.TestCase):

  def testHardMiningWithSingleLossType(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    # Uses only location loss to select hard examples
    loss_op = losses.HardExampleMiner(num_hard_examples=1,
                                      iou_threshold=0.0,
                                      loss_type='loc',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 100 + 3
    exp_cls_loss = 0 + 0
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningWithBothLossType(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    loss_op = losses.HardExampleMiner(num_hard_examples=1,
                                      iou_threshold=0.0,
                                      loss_type='both',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 80 + 0
    exp_cls_loss = 50 + 9
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testHardMiningNMS(self):
    location_losses = tf.constant([[100, 90, 80, 0],
                                   [0, 1, 2, 3]], tf.float32)
    cls_losses = tf.constant([[0, 10, 50, 110],
                              [9, 6, 3, 0]], tf.float32)
    box_corners = tf.constant([[0.1, 0.1, 0.9, 0.9],
                               [0.9, 0.9, 0.99, 0.99],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.1, 0.1, 0.9, 0.9]], tf.float32)
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    decoded_boxlist_list.append(box_list.BoxList(box_corners))
    loss_op = losses.HardExampleMiner(num_hard_examples=2,
                                      iou_threshold=0.5,
                                      loss_type='cls',
                                      cls_loss_weight=1,
                                      loc_loss_weight=1)
    (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                   decoded_boxlist_list)
    exp_loc_loss = 0 + 90 + 0 + 1
    exp_cls_loss = 110 + 10 + 9 + 6
    with self.test_session() as sess:
      loc_loss_output = sess.run(loc_loss)
      self.assertAllClose(loc_loss_output, exp_loc_loss)
      cls_loss_output = sess.run(cls_loss)
      self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testEnforceNegativesPerPositiveRatio(self):
    location_losses = tf.constant([[100, 90, 80, 0, 1, 2,
                                    3, 10, 20, 100, 20, 3]], tf.float32)
    cls_losses = tf.constant([[0, 0, 100, 0, 90, 70,
                               0, 60, 0, 17, 13, 0]], tf.float32)
    box_corners = tf.constant([[0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.5, 0.1],
                               [0.0, 0.0, 0.6, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.8, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 1.0, 0.1],
                               [0.0, 0.0, 1.1, 0.1],
                               [0.0, 0.0, 0.2, 0.1]], tf.float32)
    match_results = tf.constant([2, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 3])
    match_list = [matcher.Match(match_results)]
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))

    max_negatives_per_positive_list = [0.0, 0.5, 1.0, 1.5, 10]
    exp_loc_loss_list = [80 + 2,
                         80 + 1 + 2,
                         80 + 1 + 2 + 10,
                         80 + 1 + 2 + 10 + 100,
                         80 + 1 + 2 + 10 + 100 + 20]
    exp_cls_loss_list = [100 + 70,
                         100 + 90 + 70,
                         100 + 90 + 70 + 60,
                         100 + 90 + 70 + 60 + 17,
                         100 + 90 + 70 + 60 + 17 + 13]

    for max_negatives_per_positive, exp_loc_loss, exp_cls_loss in zip(
        max_negatives_per_positive_list, exp_loc_loss_list, exp_cls_loss_list):
      loss_op = losses.HardExampleMiner(
          num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
          cls_loss_weight=1, loc_loss_weight=1,
          max_negatives_per_positive=max_negatives_per_positive)
      (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                     decoded_boxlist_list, match_list)
      loss_op.summarize()

      with self.test_session() as sess:
        loc_loss_output = sess.run(loc_loss)
        self.assertAllClose(loc_loss_output, exp_loc_loss)
        cls_loss_output = sess.run(cls_loss)
        self.assertAllClose(cls_loss_output, exp_cls_loss)

  def testEnforceNegativesPerPositiveRatioWithMinNegativesPerImage(self):
    location_losses = tf.constant([[100, 90, 80, 0, 1, 2,
                                    3, 10, 20, 100, 20, 3]], tf.float32)
    cls_losses = tf.constant([[0, 0, 100, 0, 90, 70,
                               0, 60, 0, 17, 13, 0]], tf.float32)
    box_corners = tf.constant([[0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.5, 0.1],
                               [0.0, 0.0, 0.6, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 0.8, 0.1],
                               [0.0, 0.0, 0.2, 0.1],
                               [0.0, 0.0, 1.0, 0.1],
                               [0.0, 0.0, 1.1, 0.1],
                               [0.0, 0.0, 0.2, 0.1]], tf.float32)
    match_results = tf.constant([-1] * 12)
    match_list = [matcher.Match(match_results)]
    decoded_boxlist_list = []
    decoded_boxlist_list.append(box_list.BoxList(box_corners))

    min_negatives_per_image_list = [0, 1, 2, 4, 5, 6]
    exp_loc_loss_list = [0,
                         80,
                         80 + 1,
                         80 + 1 + 2 + 10,
                         80 + 1 + 2 + 10 + 100,
                         80 + 1 + 2 + 10 + 100 + 20]
    exp_cls_loss_list = [0,
                         100,
                         100 + 90,
                         100 + 90 + 70 + 60,
                         100 + 90 + 70 + 60 + 17,
                         100 + 90 + 70 + 60 + 17 + 13]

    for min_negatives_per_image, exp_loc_loss, exp_cls_loss in zip(
        min_negatives_per_image_list, exp_loc_loss_list, exp_cls_loss_list):
      loss_op = losses.HardExampleMiner(
          num_hard_examples=None, iou_threshold=0.9999, loss_type='cls',
          cls_loss_weight=1, loc_loss_weight=1,
          max_negatives_per_positive=3,
          min_negatives_per_image=min_negatives_per_image)
      (loc_loss, cls_loss) = loss_op(location_losses, cls_losses,
                                     decoded_boxlist_list, match_list)
      with self.test_session() as sess:
        loc_loss_output = sess.run(loc_loss)
        self.assertAllClose(loc_loss_output, exp_loc_loss)
        cls_loss_output = sess.run(cls_loss)
        self.assertAllClose(cls_loss_output, exp_cls_loss)


if __name__ == '__main__':
  tf.test.main()
