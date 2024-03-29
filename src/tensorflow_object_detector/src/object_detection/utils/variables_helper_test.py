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

"""Tests for object_detection.utils.variables_helper."""
import os

import tensorflow as tf

from object_detection.utils import variables_helper


class FilterVariablesTest(tf.test.TestCase):

  def _create_variables(self):
    return [tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights'),
            tf.Variable(1.0, name='FeatureExtractor/InceptionV3/biases'),
            tf.Variable(1.0, name='StackProposalGenerator/weights'),
            tf.Variable(1.0, name='StackProposalGenerator/biases')]

  def test_return_all_variables_when_empty_regex(self):
    variables = self._create_variables()
    out_variables = variables_helper.filter_variables(variables, [''])
    self.assertItemsEqual(out_variables, variables)

  def test_return_variables_which_do_not_match_single_regex(self):
    variables = self._create_variables()
    out_variables = variables_helper.filter_variables(variables,
                                                      ['FeatureExtractor/.*'])
    self.assertItemsEqual(out_variables, variables[2:])

  def test_return_variables_which_do_not_match_any_regex_in_list(self):
    variables = self._create_variables()
    out_variables = variables_helper.filter_variables(variables, [
        'FeatureExtractor.*biases', 'StackProposalGenerator.*biases'
    ])
    self.assertItemsEqual(out_variables, [variables[0], variables[2]])

  def test_return_variables_matching_empty_regex_list(self):
    variables = self._create_variables()
    out_variables = variables_helper.filter_variables(
        variables, [''], invert=True)
    self.assertItemsEqual(out_variables, [])

  def test_return_variables_matching_some_regex_in_list(self):
    variables = self._create_variables()
    out_variables = variables_helper.filter_variables(
        variables,
        ['FeatureExtractor.*biases', 'StackProposalGenerator.*biases'],
        invert=True)
    self.assertItemsEqual(out_variables, [variables[1], variables[3]])


class MultiplyGradientsMatchingRegexTest(tf.test.TestCase):

  def _create_grads_and_vars(self):
    return [(tf.constant(1.0),
             tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights')),
            (tf.constant(2.0),
             tf.Variable(2.0, name='FeatureExtractor/InceptionV3/biases')),
            (tf.constant(3.0),
             tf.Variable(3.0, name='StackProposalGenerator/weights')),
            (tf.constant(4.0),
             tf.Variable(4.0, name='StackProposalGenerator/biases'))]

  def test_multiply_all_feature_extractor_variables(self):
    grads_and_vars = self._create_grads_and_vars()
    regex_list = ['FeatureExtractor/.*']
    multiplier = 0.0
    grads_and_vars = variables_helper.multiply_gradients_matching_regex(
        grads_and_vars, regex_list, multiplier)
    exp_output = [(0.0, 1.0), (0.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      output = sess.run(grads_and_vars)
      self.assertItemsEqual(output, exp_output)

  def test_multiply_all_bias_variables(self):
    grads_and_vars = self._create_grads_and_vars()
    regex_list = ['.*/biases']
    multiplier = 0.0
    grads_and_vars = variables_helper.multiply_gradients_matching_regex(
        grads_and_vars, regex_list, multiplier)
    exp_output = [(1.0, 1.0), (0.0, 2.0), (3.0, 3.0), (0.0, 4.0)]
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      output = sess.run(grads_and_vars)
      self.assertItemsEqual(output, exp_output)


class FreezeGradientsMatchingRegexTest(tf.test.TestCase):

  def _create_grads_and_vars(self):
    return [(tf.constant(1.0),
             tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights')),
            (tf.constant(2.0),
             tf.Variable(2.0, name='FeatureExtractor/InceptionV3/biases')),
            (tf.constant(3.0),
             tf.Variable(3.0, name='StackProposalGenerator/weights')),
            (tf.constant(4.0),
             tf.Variable(4.0, name='StackProposalGenerator/biases'))]

  def test_freeze_all_feature_extractor_variables(self):
    grads_and_vars = self._create_grads_and_vars()
    regex_list = ['FeatureExtractor/.*']
    grads_and_vars = variables_helper.freeze_gradients_matching_regex(
        grads_and_vars, regex_list)
    exp_output = [(3.0, 3.0), (4.0, 4.0)]
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      output = sess.run(grads_and_vars)
      self.assertItemsEqual(output, exp_output)


class GetVariablesAvailableInCheckpointTest(tf.test.TestCase):

  def test_return_all_variables_from_checkpoint(self):
    variables = [
        tf.Variable(1.0, name='weights'),
        tf.Variable(1.0, name='biases')
    ]
    checkpoint_path = os.path.join(self.get_temp_dir(), 'graph.pb')
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(variables)
    with self.test_session() as sess:
      sess.run(init_op)
      saver.save(sess, checkpoint_path)
    out_variables = variables_helper.get_variables_available_in_checkpoint(
        variables, checkpoint_path)
    self.assertItemsEqual(out_variables, variables)

  def test_return_variables_available_in_checkpoint(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'graph.pb')
    graph1_variables = [
        tf.Variable(1.0, name='weights'),
    ]
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(graph1_variables)
    with self.test_session() as sess:
      sess.run(init_op)
      saver.save(sess, checkpoint_path)

    graph2_variables = graph1_variables + [tf.Variable(1.0, name='biases')]
    out_variables = variables_helper.get_variables_available_in_checkpoint(
        graph2_variables, checkpoint_path)
    self.assertItemsEqual(out_variables, graph1_variables)

  def test_return_variables_available_an_checkpoint_with_dict_inputs(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), 'graph.pb')
    graph1_variables = [
        tf.Variable(1.0, name='ckpt_weights'),
    ]
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(graph1_variables)
    with self.test_session() as sess:
      sess.run(init_op)
      saver.save(sess, checkpoint_path)

    graph2_variables_dict = {
        'ckpt_weights': tf.Variable(1.0, name='weights'),
        'ckpt_biases': tf.Variable(1.0, name='biases')
    }
    out_variables = variables_helper.get_variables_available_in_checkpoint(
        graph2_variables_dict, checkpoint_path)
    self.assertTrue(isinstance(out_variables, dict))
    self.assertItemsEqual(out_variables.keys(), ['ckpt_weights'])
    self.assertTrue(out_variables['ckpt_weights'].op.name == 'weights')


if __name__ == '__main__':
  tf.test.main()
