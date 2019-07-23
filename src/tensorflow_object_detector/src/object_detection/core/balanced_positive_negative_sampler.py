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

"""Class to subsample minibatches by balancing positives and negatives.

Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired batch_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired batch_size, it returns fewer examples.

The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.
"""

import tensorflow as tf

from object_detection.core import minibatch_sampler


class BalancedPositiveNegativeSampler(minibatch_sampler.MinibatchSampler):
  """Subsamples minibatches to a desired balance of positives and negatives."""

  def __init__(self, positive_fraction=0.5):
    """Constructs a minibatch sampler.

    Args:
      positive_fraction: desired fraction of positive examples (scalar in [0,1])

    Raises:
      ValueError: if positive_fraction < 0, or positive_fraction > 1
    """
    if positive_fraction < 0 or positive_fraction > 1:
      raise ValueError('positive_fraction should be in range [0,1]. '
                       'Received: %s.' % positive_fraction)
    self._positive_fraction = positive_fraction

  def subsample(self, indicator, batch_size, labels):
    """Returns subsampled minibatch.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
      batch_size: desired batch size.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
          (=False) examples.

    Returns:
      is_sampled: boolean tensor of shape [N], True for entries which are
          sampled.

    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
    if len(indicator.get_shape().as_list()) != 1:
      raise ValueError('indicator must be 1 dimensional, got a tensor of '
                       'shape %s' % indicator.get_shape())
    if len(labels.get_shape().as_list()) != 1:
      raise ValueError('labels must be 1 dimensional, got a tensor of '
                       'shape %s' % labels.get_shape())
    if labels.dtype != tf.bool:
      raise ValueError('labels should be of type bool. Received: %s' %
                       labels.dtype)
    if indicator.dtype != tf.bool:
      raise ValueError('indicator should be of type bool. Received: %s' %
                       indicator.dtype)

    # Only sample from indicated samples
    negative_idx = tf.logical_not(labels)
    positive_idx = tf.logical_and(labels, indicator)
    negative_idx = tf.logical_and(negative_idx, indicator)

    # Sample positive and negative samples separately
    max_num_pos = int(self._positive_fraction * batch_size)
    sampled_pos_idx = self.subsample_indicator(positive_idx, max_num_pos)
    max_num_neg = batch_size - tf.reduce_sum(tf.cast(sampled_pos_idx, tf.int32))
    sampled_neg_idx = self.subsample_indicator(negative_idx, max_num_neg)

    sampled_idx = tf.logical_or(sampled_pos_idx, sampled_neg_idx)
    return sampled_idx
