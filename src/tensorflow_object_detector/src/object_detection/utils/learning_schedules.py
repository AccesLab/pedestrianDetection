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

"""Library of common learning rate schedules."""

import tensorflow as tf


def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0):
  """Exponential decay schedule with burn-in period.

  In this schedule, learning rate is fixed at burnin_learning_rate
  for a fixed period, before transitioning to a regular exponential
  decay schedule.

  Args:
    global_step: int tensor representing global step.
    learning_rate_base: base learning rate.
    learning_rate_decay_steps: steps to take between decaying the learning rate.
      Note that this includes the number of burn-in steps.
    learning_rate_decay_factor: multiplicative factor by which to decay
      learning rate.
    burnin_learning_rate: initial learning rate during burn-in period.  If
      0.0 (which is the default), then the burn-in learning rate is simply
      set to learning_rate_base.
    burnin_steps: number of steps to use burnin learning rate.

  Returns:
    a (scalar) float tensor representing learning rate
  """
  if burnin_learning_rate == 0:
    burnin_learning_rate = learning_rate_base
  post_burnin_learning_rate = tf.train.exponential_decay(
      learning_rate_base,
      global_step,
      learning_rate_decay_steps,
      learning_rate_decay_factor,
      staircase=True)
  return tf.cond(
      tf.less(global_step, burnin_steps),
      lambda: tf.convert_to_tensor(burnin_learning_rate),
      lambda: post_burnin_learning_rate)


def manual_stepping(global_step, boundaries, rates):
  """Manually stepped learning rate schedule.

  This function provides fine grained control over learning rates.  One must
  specify a sequence of learning rates as well as a set of integer steps
  at which the current learning rate must transition to the next.  For example,
  if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning
  rate returned by this function is .1 for global_step=0,...,4, .01 for
  global_step=5...9, and .001 for global_step=10 and onward.

  Args:
    global_step: int64 (scalar) tensor representing global step.
    boundaries: a list of global steps at which to switch learning
      rates.  This list is assumed to consist of increasing positive integers.
    rates: a list of (float) learning rates corresponding to intervals between
      the boundaries.  The length of this list must be exactly
      len(boundaries) + 1.

  Returns:
    a (scalar) float tensor representing learning rate
  Raises:
    ValueError: if one of the following checks fails:
      1. boundaries is a strictly increasing list of positive integers
      2. len(rates) == len(boundaries) + 1
  """
  if any([b < 0 for b in boundaries]) or any(
      [not isinstance(b, int) for b in boundaries]):
    raise ValueError('boundaries must be a list of positive integers')
  if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
    raise ValueError('Entries in boundaries must be strictly increasing.')
  if any([not isinstance(r, float) for r in rates]):
    raise ValueError('Learning rates must be floats')
  if len(rates) != len(boundaries) + 1:
    raise ValueError('Number of provided learning rates must exceed '
                     'number of boundary points by exactly 1.')
  step_boundaries = tf.constant(boundaries, tf.int64)
  learning_rates = tf.constant(rates, tf.float32)
  unreached_boundaries = tf.reshape(tf.where(
      tf.greater(step_boundaries, global_step)), [-1])
  unreached_boundaries = tf.concat([unreached_boundaries, [len(boundaries)]], 0)
  index = tf.reshape(tf.reduce_min(unreached_boundaries), [1])
  return tf.reshape(tf.slice(learning_rates, index, [1]), [])
