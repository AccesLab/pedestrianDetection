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

"""Tests for object_detection.utils.learning_schedules."""
import tensorflow as tf

from object_detection.utils import learning_schedules


class LearningSchedulesTest(tf.test.TestCase):

  def testExponentialDecayWithBurnin(self):
    global_step = tf.placeholder(tf.int32, [])
    learning_rate_base = 1.0
    learning_rate_decay_steps = 3
    learning_rate_decay_factor = .1
    burnin_learning_rate = .5
    burnin_steps = 2
    exp_rates = [.5, .5, 1, .1, .1, .1, .01, .01]
    learning_rate = learning_schedules.exponential_decay_with_burnin(
        global_step, learning_rate_base, learning_rate_decay_steps,
        learning_rate_decay_factor, burnin_learning_rate, burnin_steps)
    with self.test_session() as sess:
      output_rates = []
      for input_global_step in range(8):
        output_rate = sess.run(learning_rate,
                               feed_dict={global_step: input_global_step})
        output_rates.append(output_rate)
      self.assertAllClose(output_rates, exp_rates)

  def testManualStepping(self):
    global_step = tf.placeholder(tf.int64, [])
    boundaries = [2, 3, 7]
    rates = [1.0, 2.0, 3.0, 4.0]
    exp_rates = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    learning_rate = learning_schedules.manual_stepping(global_step, boundaries,
                                                       rates)
    with self.test_session() as sess:
      output_rates = []
      for input_global_step in range(10):
        output_rate = sess.run(learning_rate,
                               feed_dict={global_step: input_global_step})
        output_rates.append(output_rate)
      self.assertAllClose(output_rates, exp_rates)

if __name__ == '__main__':
  tf.test.main()
