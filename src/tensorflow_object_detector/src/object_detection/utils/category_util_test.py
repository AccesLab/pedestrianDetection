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

"""Tests for object_detection.utils.category_util."""
import os

import tensorflow as tf

from object_detection.utils import category_util


class EvalUtilTest(tf.test.TestCase):

  def test_load_categories_from_csv_file(self):
    csv_data = """
        0,"cat"
        1,"dog"
        2,"bird"
    """.strip(' ')
    csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
    with tf.gfile.Open(csv_path, 'wb') as f:
      f.write(csv_data)

    categories = category_util.load_categories_from_csv_file(csv_path)
    self.assertTrue({'id': 0, 'name': 'cat'} in categories)
    self.assertTrue({'id': 1, 'name': 'dog'} in categories)
    self.assertTrue({'id': 2, 'name': 'bird'} in categories)

  def test_save_categories_to_csv_file(self):
    categories = [
        {'id': 0, 'name': 'cat'},
        {'id': 1, 'name': 'dog'},
        {'id': 2, 'name': 'bird'},
    ]
    csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
    category_util.save_categories_to_csv_file(categories, csv_path)
    saved_categories = category_util.load_categories_from_csv_file(csv_path)
    self.assertEqual(saved_categories, categories)


if __name__ == '__main__':
  tf.test.main()
