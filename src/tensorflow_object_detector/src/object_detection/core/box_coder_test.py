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

"""Tests for object_detection.core.box_coder."""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list


class MockBoxCoder(box_coder.BoxCoder):
  """Test BoxCoder that encodes/decodes using the multiply-by-two function."""

  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    return 2.0 * boxes.get()

  def _decode(self, rel_codes, anchors):
    return box_list.BoxList(rel_codes / 2.0)


class BoxCoderTest(tf.test.TestCase):

  def test_batch_decode(self):
    mock_anchor_corners = tf.constant(
        [[0, 0.1, 0.2, 0.3], [0.2, 0.4, 0.4, 0.6]], tf.float32)
    mock_anchors = box_list.BoxList(mock_anchor_corners)
    mock_box_coder = MockBoxCoder()

    expected_boxes = [[[0.0, 0.1, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8]],
                      [[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0]]]

    encoded_boxes_list = [mock_box_coder.encode(
        box_list.BoxList(tf.constant(boxes)), mock_anchors)
                          for boxes in expected_boxes]
    encoded_boxes = tf.stack(encoded_boxes_list)
    decoded_boxes = box_coder.batch_decode(
        encoded_boxes, mock_box_coder, mock_anchors)

    with self.test_session() as sess:
      decoded_boxes_result = sess.run(decoded_boxes)
      self.assertAllClose(expected_boxes, decoded_boxes_result)


if __name__ == '__main__':
  tf.test.main()
