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

"""Provides functions to prefetch tensors to feed into models."""
import tensorflow as tf


def prefetch(tensor_dict, capacity):
  """Creates a prefetch queue for tensors.

  Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a
  dequeue op that evaluates to a tensor_dict. This function is useful in
  prefetching preprocessed tensors so that the data is readily available for
  consumers.

  Example input pipeline when you don't need batching:
  ----------------------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)
  tensor_dict = prefetch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  ----------------------------------------------------

  For input pipelines with batching, refer to core/batcher.py

  Args:
    tensor_dict: a dictionary of tensors to prefetch.
    capacity: the size of the prefetch queue.

  Returns:
    a FIFO prefetcher queue
  """
  names = list(tensor_dict.keys())
  dtypes = [t.dtype for t in tensor_dict.values()]
  shapes = [t.get_shape() for t in tensor_dict.values()]
  prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes,
                                       shapes=shapes,
                                       names=names,
                                       name='prefetch_queue')
  enqueue_op = prefetch_queue.enqueue(tensor_dict)
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))
  tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name,
                                                      capacity),
                    tf.to_float(prefetch_queue.size()) * (1. / capacity))
  return prefetch_queue
