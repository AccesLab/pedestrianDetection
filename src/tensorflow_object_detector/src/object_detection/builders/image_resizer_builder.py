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

"""Builder function for image resizing operations."""
import functools

from object_detection.core import preprocessor
from object_detection.protos import image_resizer_pb2


def build(image_resizer_config):
  """Builds callable for image resizing operations.

  Args:
    image_resizer_config: image_resizer.proto object containing parameters for
      an image resizing operation.

  Returns:
    image_resizer_fn: Callable for image resizing.  This callable always takes
      a rank-3 image tensor (corresponding to a single image) and returns a
      rank-3 image tensor, possibly with new spatial dimensions.

  Raises:
    ValueError: if `image_resizer_config` is of incorrect type.
    ValueError: if `image_resizer_config.image_resizer_oneof` is of expected
      type.
    ValueError: if min_dimension > max_dimension when keep_aspect_ratio_resizer
      is used.
  """
  if not isinstance(image_resizer_config, image_resizer_pb2.ImageResizer):
    raise ValueError('image_resizer_config not of type '
                     'image_resizer_pb2.ImageResizer.')

  if image_resizer_config.WhichOneof(
      'image_resizer_oneof') == 'keep_aspect_ratio_resizer':
    keep_aspect_ratio_config = image_resizer_config.keep_aspect_ratio_resizer
    if not (keep_aspect_ratio_config.min_dimension
            <= keep_aspect_ratio_config.max_dimension):
      raise ValueError('min_dimension > max_dimension')
    return functools.partial(
        preprocessor.resize_to_range,
        min_dimension=keep_aspect_ratio_config.min_dimension,
        max_dimension=keep_aspect_ratio_config.max_dimension)
  if image_resizer_config.WhichOneof(
      'image_resizer_oneof') == 'fixed_shape_resizer':
    fixed_shape_resizer_config = image_resizer_config.fixed_shape_resizer
    return functools.partial(preprocessor.resize_image,
                             new_height=fixed_shape_resizer_config.height,
                             new_width=fixed_shape_resizer_config.width)
  raise ValueError('Invalid image resizer option.')
