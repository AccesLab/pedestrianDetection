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

"""Tests for losses_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import losses_builder
from object_detection.core import losses
from object_detection.protos import losses_pb2


class LocalizationLossBuilderTest(tf.test.TestCase):

  def test_build_weighted_l2_localization_loss(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, localization_loss, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(localization_loss,
                               losses.WeightedL2LocalizationLoss))

  def test_build_weighted_smooth_l1_localization_loss(self):
    losses_text_proto = """
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, localization_loss, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(localization_loss,
                               losses.WeightedSmoothL1LocalizationLoss))

  def test_build_weighted_iou_localization_loss(self):
    losses_text_proto = """
      localization_loss {
        weighted_iou {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, localization_loss, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(localization_loss,
                               losses.WeightedIOULocalizationLoss))

  def test_anchorwise_output(self):
    losses_text_proto = """
      localization_loss {
        weighted_smooth_l1 {
          anchorwise_output: true
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, localization_loss, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(localization_loss,
                               losses.WeightedSmoothL1LocalizationLoss))
    predictions = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
    targets = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
    weights = tf.constant([[1.0, 1.0]])
    loss = localization_loss(predictions, targets, weights=weights)
    self.assertEqual(loss.shape, [1, 2])

  def test_raise_error_on_empty_localization_config(self):
    losses_text_proto = """
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    with self.assertRaises(ValueError):
      losses_builder._build_localization_loss(losses_proto)


class ClassificationLossBuilderTest(tf.test.TestCase):

  def test_build_weighted_sigmoid_classification_loss(self):
    losses_text_proto = """
      classification_loss {
        weighted_sigmoid {
        }
      }
      localization_loss {
        weighted_l2 {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    classification_loss, _, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(classification_loss,
                               losses.WeightedSigmoidClassificationLoss))

  def test_build_weighted_softmax_classification_loss(self):
    losses_text_proto = """
      classification_loss {
        weighted_softmax {
        }
      }
      localization_loss {
        weighted_l2 {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    classification_loss, _, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(classification_loss,
                               losses.WeightedSoftmaxClassificationLoss))

  def test_build_bootstrapped_sigmoid_classification_loss(self):
    losses_text_proto = """
      classification_loss {
        bootstrapped_sigmoid {
          alpha: 0.5
        }
      }
      localization_loss {
        weighted_l2 {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    classification_loss, _, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(classification_loss,
                               losses.BootstrappedSigmoidClassificationLoss))

  def test_anchorwise_output(self):
    losses_text_proto = """
      classification_loss {
        weighted_sigmoid {
          anchorwise_output: true
        }
      }
      localization_loss {
        weighted_l2 {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    classification_loss, _, _, _, _ = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(classification_loss,
                               losses.WeightedSigmoidClassificationLoss))
    predictions = tf.constant([[[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]]])
    targets = tf.constant([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    weights = tf.constant([[1.0, 1.0]])
    loss = classification_loss(predictions, targets, weights=weights)
    self.assertEqual(loss.shape, [1, 2])

  def test_raise_error_on_empty_config(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    with self.assertRaises(ValueError):
      losses_builder.build(losses_proto)


class HardExampleMinerBuilderTest(tf.test.TestCase):

  def test_do_not_build_hard_example_miner_by_default(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, _, _, _, hard_example_miner = losses_builder.build(losses_proto)
    self.assertEqual(hard_example_miner, None)

  def test_build_hard_example_miner_for_classification_loss(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
      hard_example_miner {
        loss_type: CLASSIFICATION
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, _, _, _, hard_example_miner = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
    self.assertEqual(hard_example_miner._loss_type, 'cls')

  def test_build_hard_example_miner_for_localization_loss(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
      hard_example_miner {
        loss_type: LOCALIZATION
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, _, _, _, hard_example_miner = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
    self.assertEqual(hard_example_miner._loss_type, 'loc')

  def test_build_hard_example_miner_with_non_default_values(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
      hard_example_miner {
        num_hard_examples: 32
        iou_threshold: 0.5
        loss_type: LOCALIZATION
        max_negatives_per_positive: 10
        min_negatives_per_image: 3
      }
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    _, _, _, _, hard_example_miner = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
    self.assertEqual(hard_example_miner._num_hard_examples, 32)
    self.assertAlmostEqual(hard_example_miner._iou_threshold, 0.5)
    self.assertEqual(hard_example_miner._max_negatives_per_positive, 10)
    self.assertEqual(hard_example_miner._min_negatives_per_image, 3)


class LossBuilderTest(tf.test.TestCase):

  def test_build_all_loss_parameters(self):
    losses_text_proto = """
      localization_loss {
        weighted_l2 {
        }
      }
      classification_loss {
        weighted_softmax {
        }
      }
      hard_example_miner {
      }
      classification_weight: 0.8
      localization_weight: 0.2
    """
    losses_proto = losses_pb2.Loss()
    text_format.Merge(losses_text_proto, losses_proto)
    (classification_loss, localization_loss,
     classification_weight, localization_weight,
     hard_example_miner) = losses_builder.build(losses_proto)
    self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
    self.assertTrue(isinstance(classification_loss,
                               losses.WeightedSoftmaxClassificationLoss))
    self.assertTrue(isinstance(localization_loss,
                               losses.WeightedL2LocalizationLoss))
    self.assertAlmostEqual(classification_weight, 0.8)
    self.assertAlmostEqual(localization_weight, 0.2)


if __name__ == '__main__':
  tf.test.main()
