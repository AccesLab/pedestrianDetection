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

"""Abstract detection model.

This file defines a generic base class for detection models.  Programs that are
designed to work with arbitrary detection models should only depend on this
class.  We intend for the functions in this class to follow tensor-in/tensor-out
design, thus all functions have tensors or lists/dictionaries holding tensors as
inputs and outputs.

Abstractly, detection models predict output tensors given input images
which can be passed to a loss function at training time or passed to a
postprocessing function at eval time.  The computation graphs at a high level
consequently look as follows:

Training time:
inputs (images tensor) -> preprocess -> predict -> loss -> outputs (loss tensor)

Evaluation time:
inputs (images tensor) -> preprocess -> predict -> postprocess
 -> outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)

DetectionModels must thus implement four functions (1) preprocess, (2) predict,
(3) postprocess and (4) loss.  DetectionModels should make no assumptions about
the input size or aspect ratio --- they are responsible for doing any
resize/reshaping necessary (see docstring for the preprocess function).
Output classes are always integers in the range [0, num_classes).  Any mapping
of these integers to semantic labels is to be handled outside of this class.

By default, DetectionModels produce bounding box detections; However, we support
a handful of auxiliary annotations associated with each bounding box, namely,
instance masks and keypoints.
"""
from abc import ABCMeta
from abc import abstractmethod

from object_detection.core import standard_fields as fields


class DetectionModel(object):
  """Abstract base class for detection models."""
  __metaclass__ = ABCMeta

  def __init__(self, num_classes):
    """Constructor.

    Args:
      num_classes: number of classes.  Note that num_classes *does not* include
      background categories that might be implicitly be predicted in various
      implementations.
    """
    self._num_classes = num_classes
    self._groundtruth_lists = {}

  @property
  def num_classes(self):
    return self._num_classes

  def groundtruth_lists(self, field):
    """Access list of groundtruth tensors.

    Args:
      field: a string key, options are
        fields.BoxListFields.{boxes,classes,masks,keypoints}

    Returns:
      a list of tensors holding groundtruth information (see also
      provide_groundtruth function below), with one entry for each image in the
      batch.
    Raises:
      RuntimeError: if the field has not been provided via provide_groundtruth.
    """
    if field not in self._groundtruth_lists:
      raise RuntimeError('Groundtruth tensor %s has not been provided', field)
    return self._groundtruth_lists[field]

  @abstractmethod
  def preprocess(self, inputs):
    """Input preprocessing.

    To be overridden by implementations.

    This function is responsible for any scaling/shifting of input values that
    is necessary prior to running the detector on an input image.
    It is also responsible for any resizing that might be necessary as images
    are assumed to arrive in arbitrary sizes.  While this function could
    conceivably be part of the predict method (below), it is often convenient
    to keep these separate --- for example, we may want to preprocess on one
    device, place onto a queue, and let another device (e.g., the GPU) handle
    prediction.

    A few important notes about the preprocess function:
    + We assume that this operation does not have any trainable variables nor
    does it affect the groundtruth annotations in any way (thus data
    augmentation operations such as random cropping should be performed
    externally).
    + There is no assumption that the batchsize in this function is the same as
    the batch size in the predict function.  In fact, we recommend calling the
    preprocess function prior to calling any batching operations (which should
    happen outside of the model) and thus assuming that batch sizes are equal
    to 1 in the preprocess function.
    + There is also no explicit assumption that the output resolutions
    must be fixed across inputs --- this is to support "fully convolutional"
    settings in which input images can have different shapes/resolutions.

    Args:
      inputs: a [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    pass

  @abstractmethod
  def predict(self, preprocessed_inputs):
    """Predict prediction tensors from inputs tensor.

    Outputs of this function can be passed to loss or postprocess functions.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float32 tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding prediction tensors to be
        passed to the Loss or Postprocess functions.
    """
    pass

  @abstractmethod
  def postprocess(self, prediction_dict, **params):
    """Convert predicted output tensors to final detections.

    Outputs adhere to the following conventions:
    * Classes are integers in [0, num_classes); background classes are removed
      and the first non-background class is mapped to 0.
    * Boxes are to be interpreted as being in [y_min, x_min, y_max, x_max]
      format and normalized relative to the image window.
    * `num_detections` is provided for settings where detections are padded to a
      fixed number of boxes.
    * We do not specifically assume any kind of probabilistic interpretation
      of the scores --- the only important thing is their relative ordering.
      Thus implementations of the postprocess function are free to output
      logits, probabilities, calibrated probabilities, or anything else.

    Args:
      prediction_dict: a dictionary holding prediction tensors.
      **params: Additional keyword arguments for specific implementations of
        DetectionModel.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detections, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
        instance_masks: [batch, max_detections, image_height, image_width]
          (optional)
        keypoints: [batch, max_detections, num_keypoints, 2] (optional)
        num_detections: [batch]
    """
    pass

  @abstractmethod
  def loss(self, prediction_dict):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding predicted tensors

    Returns:
      a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    pass

  def provide_groundtruth(self,
                          groundtruth_boxes_list,
                          groundtruth_classes_list,
                          groundtruth_masks_list=None,
                          groundtruth_keypoints_list=None):
    """Provide groundtruth tensors.

    Args:
      groundtruth_boxes_list: a list of 2-D tf.float32 tensors of shape
        [num_boxes, 4] containing coordinates of the groundtruth boxes.
          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
          format and assumed to be normalized and clipped
          relative to the image window with y_min <= y_max and x_min <= x_max.
      groundtruth_classes_list: a list of 2-D tf.float32 one-hot (or k-hot)
        tensors of shape [num_boxes, num_classes] containing the class targets
        with the 0th index assumed to map to the first non-background class.
      groundtruth_masks_list: a list of 2-D tf.float32 tensors of
        shape [max_detections, height_in, width_in] containing instance
        masks with values in {0, 1}.  If None, no masks are provided.
        Mask resolution `height_in`x`width_in` must agree with the resolution
        of the input image tensor provided to the `preprocess` function.
      groundtruth_keypoints_list: a list of 2-D tf.float32 tensors of
        shape [batch, max_detections, num_keypoints, 2] containing keypoints.
        Keypoints are assumed to be provided in normalized coordinates and
        missing keypoints should be encoded as NaN.
    """
    self._groundtruth_lists[fields.BoxListFields.boxes] = groundtruth_boxes_list
    self._groundtruth_lists[
        fields.BoxListFields.classes] = groundtruth_classes_list
    if groundtruth_masks_list:
      self._groundtruth_lists[
          fields.BoxListFields.masks] = groundtruth_masks_list
    if groundtruth_keypoints_list:
      self._groundtruth_lists[
          fields.BoxListFields.keypoints] = groundtruth_keypoints_list

  @abstractmethod
  def restore_fn(self, checkpoint_path, from_detection_checkpoint=True):
    """Return callable for loading a foreign checkpoint into tensorflow graph.

    Loads variables from a different tensorflow graph (typically feature
    extractor variables). This enables the model to initialize based on weights
    from another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Args:
      checkpoint_path: path to checkpoint to restore.
      from_detection_checkpoint: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.

    Returns:
      a callable which takes a tf.Session as input and loads a checkpoint when
        run.
    """
    pass
