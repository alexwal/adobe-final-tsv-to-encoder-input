# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for a dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

_FILE_PATTERN = FLAGS.embeddings_dir + '-shard-*-of-*' # % (shard, num_shards)

_ITEMS_TO_DESCRIPTIONS = {
    'embedding': 'A fixed size visual semantic embedding of an image.',
    'label': 'A single integer, the unique key of the image.',
    'filename': 'A string containing the image filename for reference.'
}

def get_dataset(dataset_dir):
  # (NOTE: Instead of get_split(...) associated with other datasets.)
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  reader = tf.TFRecordReader

  keys_to_features = {
      'embedding': tf.FixedLenFeature([FLAGS.embedding_size], tf.float32),
      'label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'filename': tf.FixedLenFeature((), tf.string, default_value='')
  }

  items_to_handlers = {
      'embedding': slim.tfexample_decoder.Tensor('embedding'),
      'label': slim.tfexample_decoder.Tensor('label'),
      'filename': slim.tfexample_decoder.Tensor('filename')
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder( # NOTE: modified TFExampleDecoder to save filenames
      keys_to_features, items_to_handlers)

  num_samples = np.load(FLAGS.num_samples_dict)[()]['num_samples']

  return slim.dataset.Dataset(
      data_sources=os.path.join(dataset_dir, _FILE_PATTERN),
      reader=reader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

