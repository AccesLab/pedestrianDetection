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

"""Tests for image.understanding.object_detection.core.visualization_utils.

Testing with visualization in the following colab:
https://drive.google.com/a/google.com/file/d/0B5HnKS_hMsNARERpU3MtU3I5RFE/view?usp=sharing

"""


import numpy as np
import PIL.Image as Image
import tensorflow as tf

from object_detection.utils import visualization_utils


class VisualizationUtilsTest(tf.test.TestCase):

  def create_colorful_test_image(self):
    """This function creates an image that can be used to test vis functions.

    It makes an image composed of four colored rectangles.

    Returns:
      colorful test numpy array image.
    """
    ch255 = np.full([100, 200, 1], 255, dtype=np.uint8)
    ch128 = np.full([100, 200, 1], 128, dtype=np.uint8)
    ch0 = np.full([100, 200, 1], 0, dtype=np.uint8)
    imr = np.concatenate((ch255, ch128, ch128), axis=2)
    img = np.concatenate((ch255, ch255, ch0), axis=2)
    imb = np.concatenate((ch255, ch0, ch255), axis=2)
    imw = np.concatenate((ch128, ch128, ch128), axis=2)
    imu = np.concatenate((imr, img), axis=1)
    imd = np.concatenate((imb, imw), axis=1)
    image = np.concatenate((imu, imd), axis=0)
    return image

  def test_draw_bounding_box_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image(test_image, ymin, xmin, ymax,
                                                   xmax)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_box_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    ymin = 0.25
    ymax = 0.75
    xmin = 0.4
    xmax = 0.6

    visualization_utils.draw_bounding_box_on_image_array(
        test_image, ymin, xmin, ymax, xmax)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image(test_image, boxes)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_bounding_boxes_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    boxes = np.array([[0.25, 0.75, 0.4, 0.6],
                      [0.1, 0.1, 0.9, 0.9]])

    visualization_utils.draw_bounding_boxes_on_image_array(test_image, boxes)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_keypoints_on_image(self):
    test_image = self.create_colorful_test_image()
    test_image = Image.fromarray(test_image)
    width_original, height_original = test_image.size
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image(test_image, keypoints)
    width_final, height_final = test_image.size

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_keypoints_on_image_array(self):
    test_image = self.create_colorful_test_image()
    width_original = test_image.shape[0]
    height_original = test_image.shape[1]
    keypoints = [[0.25, 0.75], [0.4, 0.6], [0.1, 0.1], [0.9, 0.9]]

    visualization_utils.draw_keypoints_on_image_array(test_image, keypoints)
    width_final = test_image.shape[0]
    height_final = test_image.shape[1]

    self.assertEqual(width_original, width_final)
    self.assertEqual(height_original, height_final)

  def test_draw_mask_on_image_array(self):
    test_image = np.asarray([[[0, 0, 0], [0, 0, 0]],
                             [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    mask = np.asarray([[0.0, 1.0],
                       [1.0, 1.0]], dtype=np.float32)
    expected_result = np.asarray([[[0, 0, 0], [0, 0, 127]],
                                  [[0, 0, 127], [0, 0, 127]]], dtype=np.uint8)
    visualization_utils.draw_mask_on_image_array(test_image, mask,
                                                 color='Blue', alpha=.5)
    self.assertAllEqual(test_image, expected_result)


if __name__ == '__main__':
  tf.test.main()
