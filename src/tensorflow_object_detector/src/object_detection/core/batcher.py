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
"""Provides functions to batch a dictionary of input tensors."""
import collections

import tensorflow as tf

from object_detection.core import prefetcher

rt_shape_str = '_runtime_shapes'


class BatchQueue(object):
  """BatchQueue class.

  This class creates a batch queue to asynchronously enqueue tensors_dict.
  It also adds a FIFO prefetcher so that the batches are readily available
  for the consumers.  Dequeue ops for a BatchQueue object can be created via
  the Dequeue method which evaluates to a batch of tensor_dict.

  Example input pipeline with batching:
  ------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  batch_queue = batcher.BatchQueue(tensor_dict,
                                   batch_size=32,
                                   batch_queue_capacity=2000,
                                   num_batch_queue_threads=8,
                                   prefetch_queue_capacity=20)
  tensor_dict = batch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  -----------------------------------

  Notes:
  -----
  This class batches tensors of unequal sizes by zero padding and unpadding
  them after generating a batch. This can be computationally expensive when
  batching tensors (such as images) that are of vastly different sizes. So it is
  recommended that the shapes of such tensors be fully defined in tensor_dict
  while other lightweight tensors such as bounding box corners and class labels
  can be of varying sizes. Use either crop or resize operations to fully define
  the shape of an image in tensor_dict.

  It is also recommended to perform any preprocessing operations on tensors
  before passing to BatchQueue and subsequently calling the Dequeue method.

  Another caveat is that this class does not read the last batch if it is not
  full. The current implementation makes it hard to support that use case. So,
  for evaluation, when it is critical to run all the examples through your
  network use the input pipeline example mentioned in core/prefetcher.py.
  """

  def __init__(self, tensor_dict, batch_size, batch_queue_capacity,
               num_batch_queue_threads, prefetch_queue_capacity):
    """Constructs a batch queue holding tensor_dict.

    Args:
      tensor_dict: dictionary of tensors to batch.
      batch_size: batch size.
      batch_queue_capacity: max capacity of the queue from which the tensors are
        batched.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: max capacity of the queue used to prefetch
        assembled batches.
    """
    # Remember static shapes to set shapes of batched tensors.
    static_shapes = collections.OrderedDict(
        {key: tensor.get_shape() for key, tensor in tensor_dict.items()})
    # Remember runtime shapes to unpad tensors after batching.
    runtime_shapes = collections.OrderedDict(
        {(key + rt_shape_str): tf.shape(tensor)
         for key, tensor in tensor_dict.items()})

    all_tensors = tensor_dict
    all_tensors.update(runtime_shapes)
    batched_tensors = tf.train.batch(
        all_tensors,
        capacity=batch_queue_capacity,
        batch_size=batch_size,
        dynamic_pad=True,
        num_threads=num_batch_queue_threads)

    self._queue = prefetcher.prefetch(batched_tensors,
                                      prefetch_queue_capacity)
    self._static_shapes = static_shapes
    self._batch_size = batch_size

  def dequeue(self):
    """Dequeues a batch of tensor_dict from the BatchQueue.

    TODO: use allow_smaller_final_batch to allow running over the whole eval set

    Returns:
      A list of tensor_dicts of the requested batch_size.
    """
    batched_tensors = self._queue.dequeue()
    # Separate input tensors from tensors containing their runtime shapes.
    tensors = {}
    shapes = {}
    for key, batched_tensor in batched_tensors.items():
      unbatched_tensor_list = tf.unstack(batched_tensor)
      for i, unbatched_tensor in enumerate(unbatched_tensor_list):
        if rt_shape_str in key:
          shapes[(key[:-len(rt_shape_str)], i)] = unbatched_tensor
        else:
          tensors[(key, i)] = unbatched_tensor

    # Undo that padding using shapes and create a list of size `batch_size` that
    # contains tensor dictionaries.
    tensor_dict_list = []
    batch_size = self._batch_size
    for batch_id in range(batch_size):
      tensor_dict = {}
      for key in self._static_shapes:
        tensor_dict[key] = tf.slice(tensors[(key, batch_id)],
                                    tf.zeros_like(shapes[(key, batch_id)]),
                                    shapes[(key, batch_id)])
        tensor_dict[key].set_shape(self._static_shapes[key])
      tensor_dict_list.append(tensor_dict)

    return tensor_dict_list
