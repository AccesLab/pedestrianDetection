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

"""Tests for object_detection.utils.per_image_evaluation."""

import numpy as np
import tensorflow as tf

from object_detection.utils import per_image_evaluation


class SingleClassTpFpWithDifficultBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
        nms_max_output_boxes)

    self.detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                   dtype=float)
    self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=float)
    self.groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 10, 10]],
                                      dtype=float)

  def test_match_to_not_difficult_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array([False, True],
                                                         dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_match_to_difficult_box(self):
    groundtruth_groundtruth_is_difficult_list = np.array([True, False],
                                                         dtype=bool)
    scores, tp_fp_labels = self.eval._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, self.groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class SingleClassTpFpNoDifficultBoxesTest(tf.test.TestCase):

  def setUp(self):
    num_groundtruth_classes = 1
    matching_iou_threshold1 = 0.5
    matching_iou_threshold2 = 0.1
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    self.eval1 = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold1, nms_iou_threshold,
        nms_max_output_boxes)

    self.eval2 = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes, matching_iou_threshold2, nms_iou_threshold,
        nms_max_output_boxes)

    self.detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                   dtype=float)
    self.detected_scores = np.array([0.6, 0.8, 0.5], dtype=float)

  def test_no_true_positives(self):
    groundtruth_boxes = np.array([[100, 100, 105, 105]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_one_true_positives_with_large_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_one_true_positives_with_very_small_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(1, dtype=bool)
    scores, tp_fp_labels = self.eval2._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([True, False, False], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))

  def test_two_true_positives_with_large_iou_threshold(self):
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3.5, 3.5]], dtype=float)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=bool)
    scores, tp_fp_labels = self.eval1._compute_tp_fp_for_single_class(
        self.detected_boxes, self.detected_scores, groundtruth_boxes,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = np.array([0.8, 0.6, 0.5], dtype=float)
    expected_tp_fp_labels = np.array([False, True, True], dtype=bool)
    self.assertTrue(np.allclose(expected_scores, scores))
    self.assertTrue(np.allclose(expected_tp_fp_labels, tp_fp_labels))


class MultiClassesTpFpTest(tf.test.TestCase):

  def test_tp_fp(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [10, 10, 5, 5], [0, 0, 2, 2],
                               [5, 10, 10, 5], [10, 5, 5, 10], [0, 0, 3, 3]],
                              dtype=float)
    detected_scores = np.array([0.8, 0.1, 0.8, 0.9, 0.7, 0.8], dtype=float)
    detected_class_labels = np.array([0, 1, 1, 2, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3.5, 3.5]], dtype=float)
    groundtruth_class_labels = np.array([0, 2], dtype=int)
    groundtruth_groundtruth_is_difficult_list = np.zeros(2, dtype=float)
    scores, tp_fp_labels, _ = eval1.compute_object_detection_metrics(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels,
        groundtruth_groundtruth_is_difficult_list)
    expected_scores = [np.array([0.8], dtype=float)] * 3
    expected_tp_fp_labels = [np.array([True]), np.array([False]), np.array([True
                                                                           ])]
    for i in range(len(expected_scores)):
      self.assertTrue(np.allclose(expected_scores[i], scores[i]))
      self.assertTrue(np.array_equal(expected_tp_fp_labels[i], tp_fp_labels[i]))


class CorLocTest(tf.test.TestCase):

  def test_compute_corloc_with_normal_iou_threshold(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.5
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3],
                               [0, 0, 5, 5]], dtype=float)
    detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=float)
    detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 6, 6]],
                                 dtype=float)
    groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

    is_class_correctly_detected_in_image = eval1._compute_cor_loc(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels)
    expected_result = np.array([1, 0, 1], dtype=int)
    self.assertTrue(np.array_equal(expected_result,
                                   is_class_correctly_detected_in_image))

  def test_compute_corloc_with_very_large_iou_threshold(self):
    num_groundtruth_classes = 3
    matching_iou_threshold = 0.9
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000
    eval1 = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes,
                                                    matching_iou_threshold,
                                                    nms_iou_threshold,
                                                    nms_max_output_boxes)
    detected_boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3],
                               [0, 0, 5, 5]], dtype=float)
    detected_scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=float)
    detected_class_labels = np.array([0, 1, 0, 2], dtype=int)
    groundtruth_boxes = np.array([[0, 0, 1, 1], [0, 0, 3, 3], [0, 0, 6, 6]],
                                 dtype=float)
    groundtruth_class_labels = np.array([0, 0, 2], dtype=int)

    is_class_correctly_detected_in_image = eval1._compute_cor_loc(
        detected_boxes, detected_scores, detected_class_labels,
        groundtruth_boxes, groundtruth_class_labels)
    expected_result = np.array([1, 0, 0], dtype=int)
    self.assertTrue(np.array_equal(expected_result,
                                   is_class_correctly_detected_in_image))


if __name__ == '__main__':
  tf.test.main()
