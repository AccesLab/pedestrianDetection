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

"""Tests for object_detection.core.prefetcher."""
import tensorflow as tf

from object_detection.core import prefetcher

slim = tf.contrib.slim


class PrefetcherTest(tf.test.TestCase):

  def test_prefetch_tensors_with_fully_defined_shapes(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5
      examples = tf.Variable(tf.constant(0, dtype=tf.int64))
      counter = examples.count_up_to(num_batches)
      image = tf.random_normal([batch_size, image_size,
                                image_size, 3],
                               dtype=tf.float32,
                               name='images')
      label = tf.random_uniform([batch_size, 1], 0, 10,
                                dtype=tf.int32, name='labels')

      prefetch_queue = prefetcher.prefetch(tensor_dict={'counter': counter,
                                                        'image': image,
                                                        'label': label},
                                           capacity=100)
      tensor_dict = prefetch_queue.dequeue()

      self.assertAllEqual(tensor_dict['image'].get_shape().as_list(),
                          [batch_size, image_size, image_size, 3])
      self.assertAllEqual(tensor_dict['label'].get_shape().as_list(),
                          [batch_size, 1])

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        for _ in range(num_batches):
          results = sess.run(tensor_dict)
          self.assertEquals(results['image'].shape,
                            (batch_size, image_size, image_size, 3))
          self.assertEquals(results['label'].shape, (batch_size, 1))
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(tensor_dict)

  def test_prefetch_tensors_with_partially_defined_shapes(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5
      examples = tf.Variable(tf.constant(0, dtype=tf.int64))
      counter = examples.count_up_to(num_batches)
      image = tf.random_normal([batch_size,
                                tf.Variable(image_size),
                                tf.Variable(image_size), 3],
                               dtype=tf.float32,
                               name='image')
      image.set_shape([batch_size, None, None, 3])
      label = tf.random_uniform([batch_size, tf.Variable(1)], 0,
                                10, dtype=tf.int32, name='label')
      label.set_shape([batch_size, None])

      prefetch_queue = prefetcher.prefetch(tensor_dict={'counter': counter,
                                                        'image': image,
                                                        'label': label},
                                           capacity=100)
      tensor_dict = prefetch_queue.dequeue()

      self.assertAllEqual(tensor_dict['image'].get_shape().as_list(),
                          [batch_size, None, None, 3])
      self.assertAllEqual(tensor_dict['label'].get_shape().as_list(),
                          [batch_size, None])

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        for _ in range(num_batches):
          results = sess.run(tensor_dict)
          self.assertEquals(results['image'].shape,
                            (batch_size, image_size, image_size, 3))
          self.assertEquals(results['label'].shape, (batch_size, 1))
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(tensor_dict)


if __name__ == '__main__':
  tf.test.main()
