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

"""Tests for region_similarity_calculator."""
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import region_similarity_calculator


class RegionSimilarityCalculatorTest(tf.test.TestCase):

  def test_get_correct_pairwise_similarity_based_on_iou(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    exp_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    iou_similarity_calculator = region_similarity_calculator.IouSimilarity()
    iou_similarity = iou_similarity_calculator.compare(boxes1, boxes2)
    with self.test_session() as sess:
      iou_output = sess.run(iou_similarity)
      self.assertAllClose(iou_output, exp_output)

  def test_get_correct_pairwise_similarity_based_on_squared_distances(self):
    corners1 = tf.constant([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 2.0]])
    corners2 = tf.constant([[3.0, 4.0, 1.0, 0.0],
                            [-4.0, 0.0, 0.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0]])
    exp_output = [[-26, -25, 0], [-18, -27, -6]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    dist_similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
    dist_similarity = dist_similarity_calc.compare(boxes1, boxes2)
    with self.test_session() as sess:
      dist_output = sess.run(dist_similarity)
      self.assertAllClose(dist_output, exp_output)

  def test_get_correct_pairwise_similarity_based_on_ioa(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    exp_output_1 = [[2.0 / 12.0, 0, 6.0 / 400.0],
                    [1.0 / 12.0, 0.0, 5.0 / 400.0]]
    exp_output_2 = [[2.0 / 6.0, 1.0 / 5.0],
                    [0, 0],
                    [6.0 / 6.0, 5.0 / 5.0]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    ioa_similarity_calculator = region_similarity_calculator.IoaSimilarity()
    ioa_similarity_1 = ioa_similarity_calculator.compare(boxes1, boxes2)
    ioa_similarity_2 = ioa_similarity_calculator.compare(boxes2, boxes1)
    with self.test_session() as sess:
      iou_output_1, iou_output_2 = sess.run(
          [ioa_similarity_1, ioa_similarity_2])
      self.assertAllClose(iou_output_1, exp_output_1)
      self.assertAllClose(iou_output_2, exp_output_2)


if __name__ == '__main__':
  tf.test.main()
