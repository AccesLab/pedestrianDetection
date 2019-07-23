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

"""Functions for importing/exporting Object Detection categories."""
import csv

import tensorflow as tf


def load_categories_from_csv_file(csv_path):
  """Loads categories from a csv file.

  The CSV file should have one comma delimited numeric category id and string
  category name pair per line. For example:

  0,"cat"
  1,"dog"
  2,"bird"
  ...

  Args:
    csv_path: Path to the csv file to be parsed into categories.
  Returns:
    categories: A list of dictionaries representing all possible categories.
                The categories will contain an integer 'id' field and a string
                'name' field.
  Raises:
    ValueError: If the csv file is incorrectly formatted.
  """
  categories = []

  with tf.gfile.Open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
      if not row:
        continue

      if len(row) != 2:
        raise ValueError('Expected 2 fields per row in csv: %s' % ','.join(row))

      category_id = int(row[0])
      category_name = row[1]
      categories.append({'id': category_id, 'name': category_name})

  return categories


def save_categories_to_csv_file(categories, csv_path):
  """Saves categories to a csv file.

  Args:
    categories: A list of dictionaries representing categories to save to file.
                Each category must contain an 'id' and 'name' field.
    csv_path: Path to the csv file to be parsed into categories.
  """
  categories.sort(key=lambda x: x['id'])
  with tf.gfile.Open(csv_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"')
    for category in categories:
      writer.writerow([category['id'], category['name']])
