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

"""Functions to build DetectionModel training optimizers."""

import tensorflow as tf
from object_detection.utils import learning_schedules

slim = tf.contrib.slim


def build(optimizer_config, global_summaries):
  """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.
    global_summaries: A set to attach learning rate summary to.

  Returns:
    An optimizer.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')
  optimizer = None

  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    optimizer = tf.train.RMSPropOptimizer(
        _create_learning_rate(config.learning_rate, global_summaries),
        decay=config.decay,
        momentum=config.momentum_optimizer_value,
        epsilon=config.epsilon)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    optimizer = tf.train.MomentumOptimizer(
        _create_learning_rate(config.learning_rate, global_summaries),
        momentum=config.momentum_optimizer_value)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    optimizer = tf.train.AdamOptimizer(
        _create_learning_rate(config.learning_rate, global_summaries))

  if optimizer is None:
    raise ValueError('Optimizer %s not supported.' % optimizer_type)

  if optimizer_config.use_moving_average:
    optimizer = tf.contrib.opt.MovingAverageOptimizer(
        optimizer, average_decay=optimizer_config.moving_average_decay)

  return optimizer


def _create_learning_rate(learning_rate_config, global_summaries):
  """Create optimizer learning rate based on config.

  Args:
    learning_rate_config: A LearningRate proto message.
    global_summaries: A set to attach learning rate summary to.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  learning_rate = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    learning_rate = config.learning_rate

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    learning_rate = tf.train.exponential_decay(
        config.initial_learning_rate,
        slim.get_or_create_global_step(),
        config.decay_steps,
        config.decay_factor,
        staircase=config.staircase)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    learning_rate = learning_schedules.manual_stepping(
        slim.get_or_create_global_step(), learning_rate_step_boundaries,
        learning_rate_sequence)

  if learning_rate is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  global_summaries.add(tf.summary.scalar('Learning Rate', learning_rate))
  return learning_rate
