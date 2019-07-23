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

"""Helper functions to access TensorShape values.

The rank 4 tensor_shape must be of the form [batch_size, height, width, depth].
"""


def get_batch_size(tensor_shape):
  """Returns batch size from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the batch size of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[0].value


def get_height(tensor_shape):
  """Returns height from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the height of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[1].value


def get_width(tensor_shape):
  """Returns width from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the width of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[2].value


def get_depth(tensor_shape):
  """Returns depth from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the depth of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[3].value
