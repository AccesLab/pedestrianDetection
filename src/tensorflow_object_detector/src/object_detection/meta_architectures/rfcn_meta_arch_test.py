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

"""Tests for object_detection.meta_architectures.rfcn_meta_arch."""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib
from object_detection.meta_architectures import rfcn_meta_arch


class RFCNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase):

  def _get_second_stage_box_predictor_text_proto(self):
    box_predictor_text_proto = """
      rfcn_box_predictor {
        conv_hyperparams {
          op: CONV
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0005
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    """
    return box_predictor_text_proto

  def _get_model(self, box_predictor, **common_kwargs):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=box_predictor, **common_kwargs)


if __name__ == '__main__':
  tf.test.main()
