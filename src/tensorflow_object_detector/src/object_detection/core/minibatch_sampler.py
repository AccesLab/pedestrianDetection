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
# ==============================================================================

"""Base minibatch sampler module.

The job of the minibatch_sampler is to subsample a minibatch based on some
criterion.

The main function call is:
    subsample(indicator, batch_size, **params).
Indicator is a 1d boolean tensor where True denotes which examples can be
sampled. It returns a boolean indicator where True denotes an example has been
sampled..

Subclasses should implement the Subsample function and can make use of the
@staticmethod SubsampleIndicator.
"""

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from object_detection.utils import ops


class MinibatchSampler(object):
  """Abstract base class for subsampling minibatches."""
  __metaclass__ = ABCMeta

  def __init__(self):
    """Constructs a minibatch sampler."""
    pass

  @abstractmethod
  def subsample(self, indicator, batch_size, **params):
    """Returns subsample of entries in indicator.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
      batch_size: desired batch size.
      **params: additional keyword arguments for specific implementations of
          the MinibatchSampler.

    Returns:
      sample_indicator: boolean tensor of shape [N] whose True entries have been
      sampled. If sum(indicator) >= batch_size, sum(is_sampled) = batch_size
    """
    pass

  @staticmethod
  def subsample_indicator(indicator, num_samples):
    """Subsample indicator vector.

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Args:
      indicator: a 1-dimensional boolean tensor indicating which elements
        are allowed to be sampled and which are not.
      num_samples: int32 scalar tensor

    Returns:
      a boolean tensor with the same shape as input (indicator) tensor
    """
    indices = tf.where(indicator)
    indices = tf.random_shuffle(indices)
    indices = tf.reshape(indices, [-1])

    num_samples = tf.minimum(tf.size(indices), num_samples)
    selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))

    selected_indicator = ops.indices_to_dense_vector(selected_indices,
                                                     tf.shape(indicator)[0])

    return tf.equal(selected_indicator, 1)
