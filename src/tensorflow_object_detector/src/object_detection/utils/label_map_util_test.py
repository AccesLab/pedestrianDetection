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
"""Tests for object_detection.utils.label_map_util."""

import os
import tensorflow as tf

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import label_map_util


class LabelMapUtilTest(tf.test.TestCase):

  def _generate_label_map(self, num_classes):
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()
    for i in range(1, num_classes + 1):
      item = label_map_proto.item.add()
      item.id = i
      item.name = 'label_' + str(i)
      item.display_name = str(i)
    return label_map_proto

  def test_get_label_map_dict(self):
    label_map_string = """
      item {
        id:2
        name:'cat'
      }
      item {
        id:1
        name:'dog'
      }
    """
    label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.gfile.Open(label_map_path, 'wb') as f:
      f.write(label_map_string)

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    self.assertEqual(label_map_dict['dog'], 1)
    self.assertEqual(label_map_dict['cat'], 2)

  def test_load_bad_label_map(self):
    label_map_string = """
      item {
        id:0
        name:'class that should not be indexed at zero'
      }
      item {
        id:2
        name:'cat'
      }
      item {
        id:1
        name:'dog'
      }
    """
    label_map_path = os.path.join(self.get_temp_dir(), 'label_map.pbtxt')
    with tf.gfile.Open(label_map_path, 'wb') as f:
      f.write(label_map_string)

    with self.assertRaises(ValueError):
      label_map_util.load_labelmap(label_map_path)

    def test_keep_categories_with_unique_id(self):
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()
    label_map_string = """
      item {
        id:2
        name:'cat'
      }
      item {
        id:1
        name:'child'
      }
      item {
        id:1
        name:'person'
      }
      item {
        id:1
        name:'n00007846'
      }
    """
    text_format.Merge(label_map_string, label_map_proto)
    categories = label_map_util.convert_label_map_to_categories(
        label_map_proto, max_num_classes=3)
    self.assertListEqual([{
        'id': 2,
        'name': u'cat'
    }, {
        'id': 1,
        'name': u'child'
    }], categories)

  def test_convert_label_map_to_categories_no_label_map(self):
    categories = label_map_util.convert_label_map_to_categories(
        None, max_num_classes=3)
    expected_categories_list = [{
        'name': u'category_1',
        'id': 1
    }, {
        'name': u'category_2',
        'id': 2
    }, {
        'name': u'category_3',
        'id': 3
    }]
    self.assertListEqual(expected_categories_list, categories)

  def test_convert_label_map_to_coco_categories(self):
    label_map_proto = self._generate_label_map(num_classes=4)
    categories = label_map_util.convert_label_map_to_categories(
        label_map_proto, max_num_classes=3)
    expected_categories_list = [{
        'name': u'1',
        'id': 1
    }, {
        'name': u'2',
        'id': 2
    }, {
        'name': u'3',
        'id': 3
    }]
    self.assertListEqual(expected_categories_list, categories)

  def test_convert_label_map_to_coco_categories_with_few_classes(self):
    label_map_proto = self._generate_label_map(num_classes=4)
    cat_no_offset = label_map_util.convert_label_map_to_categories(
        label_map_proto, max_num_classes=2)
    expected_categories_list = [{
        'name': u'1',
        'id': 1
    }, {
        'name': u'2',
        'id': 2
    }]
    self.assertListEqual(expected_categories_list, cat_no_offset)

  def test_create_category_index(self):
    categories = [{'name': u'1', 'id': 1}, {'name': u'2', 'id': 2}]
    category_index = label_map_util.create_category_index(categories)
    self.assertDictEqual({
        1: {
            'name': u'1',
            'id': 1
        },
        2: {
            'name': u'2',
            'id': 2
        }
    }, category_index)


if __name__ == '__main__':
  tf.test.main()
