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

"""A function to build localization and classification losses from config."""

from object_detection.core import losses
from object_detection.protos import losses_pb2


def build(loss_config):
  """Build losses based on the config.

  Builds classification, localization losses and optionally a hard example miner
  based on the config.

  Args:
    loss_config: A losses_pb2.Loss object.

  Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.
  """
  classification_loss = _build_classification_loss(
      loss_config.classification_loss)
  localization_loss = _build_localization_loss(
      loss_config.localization_loss)
  classification_weight = loss_config.classification_weight
  localization_weight = loss_config.localization_weight
  hard_example_miner = None
  if loss_config.HasField('hard_example_miner'):
    hard_example_miner = build_hard_example_miner(
        loss_config.hard_example_miner,
        classification_weight,
        localization_weight)
  return (classification_loss, localization_loss,
          classification_weight,
          localization_weight, hard_example_miner)


def build_hard_example_miner(config,
                             classification_weight,
                             localization_weight):
  """Builds hard example miner based on the config.

  Args:
    config: A losses_pb2.HardExampleMiner object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.

  Returns:
    Hard example miner.

  """
  loss_type = None
  if config.loss_type == losses_pb2.HardExampleMiner.BOTH:
    loss_type = 'both'
  if config.loss_type == losses_pb2.HardExampleMiner.CLASSIFICATION:
    loss_type = 'cls'
  if config.loss_type == losses_pb2.HardExampleMiner.LOCALIZATION:
    loss_type = 'loc'

  max_negatives_per_positive = None
  num_hard_examples = None
  if config.max_negatives_per_positive > 0:
    max_negatives_per_positive = config.max_negatives_per_positive
  if config.num_hard_examples > 0:
    num_hard_examples = config.num_hard_examples
  hard_example_miner = losses.HardExampleMiner(
      num_hard_examples=num_hard_examples,
      iou_threshold=config.iou_threshold,
      loss_type=loss_type,
      cls_loss_weight=classification_weight,
      loc_loss_weight=localization_weight,
      max_negatives_per_positive=max_negatives_per_positive,
      min_negatives_per_image=config.min_negatives_per_image)
  return hard_example_miner


def _build_localization_loss(loss_config):
  """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.LocalizationLoss):
    raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

  loss_type = loss_config.WhichOneof('localization_loss')

  if loss_type == 'weighted_l2':
    config = loss_config.weighted_l2
    return losses.WeightedL2LocalizationLoss(
        anchorwise_output=config.anchorwise_output)

  if loss_type == 'weighted_smooth_l1':
    config = loss_config.weighted_smooth_l1
    return losses.WeightedSmoothL1LocalizationLoss(
        anchorwise_output=config.anchorwise_output)

  if loss_type == 'weighted_iou':
    return losses.WeightedIOULocalizationLoss()

  raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
  """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    config = loss_config.weighted_sigmoid
    return losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=config.anchorwise_output)

  if loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        anchorwise_output=config.anchorwise_output)

  if loss_type == 'bootstrapped_sigmoid':
    config = loss_config.bootstrapped_sigmoid
    return losses.BootstrappedSigmoidClassificationLoss(
        alpha=config.alpha,
        bootstrap_type=('hard' if config.hard_bootstrap else 'soft'),
        anchorwise_output=config.anchorwise_output)

  raise ValueError('Empty loss config.')
