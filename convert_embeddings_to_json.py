# 6m_app.py

import logging
import random
import time
import os
import queue
import threading
import progressbar

import numpy as np

import sys

# Main slim library
from tensorflow.contrib import slim
import tensorflow as tf
from nets import inception
from preprocessing import inception_preprocessing

# Custom
from FINAL import utils
import embeddings_6m_dataset as dataset_6m
import load_6m_dataset
import s3_utils

FLAGS = tf.app.flags.FLAGS

#####


# Helper functions
import json
import base64
import gzip

def encode_feature(feature):
    # Args: np.array `feature` becomes a dimensionless FLAT np.float32 numpy
    # array if not already, padded to (next) power of 2.
    POW_OF_2 = 512
    feature = feature.flatten()
    feature = np.pad(feature, pad_width=(0, POW_OF_2 - len(feature)),
                              mode='constant').astype(np.float32)
    zipped = gzip.compress(feature.tobytes())
    return base64.b64encode(zipped).decode()

def format_row(label, emb):
  # JSON row format before training encoder
  row = {'cid' : int(label), 'descriptor' : encode_feature(emb)}
  return row

def run():
  # Load all embeddings and paths on which to run search queries
  json_dest = FLAGS.json_dest
  dataset_dir = FLAGS.embeddings_dir

  with tf.Graph().as_default():

    dataset = dataset_6m.get_dataset(dataset_dir)
    embs, labels, filenames = load_6m_dataset.load_batch(dataset, batch_size=8, num_epochs=1)
    sv = tf.train.Supervisor(logdir=None, summary_op=None)

    JSON_ROWS = []

    with sv.managed_session('') as sess:
      i = 0
      while not sv.should_stop():
        np_embs, np_labels, np_filenames = sess.run([embs, labels, filenames])
        for label, emb in zip(np_labels, np_embs):
          row = format_row(label, emb)
          JSON_ROWS.append(row)
        if i % 1000 == 0:
          print('\rLoaded %d batches...' % i, end='')
        i += 1
    print()

    print('Writing JSON (%d rows) to %s...' % (len(JSON_ROWS), json_dest))
    with open(json_dest, 'w') as f:
      for row in JSON_ROWS:
        json_string = json.dumps(row)
        f.write(json_string + '\n')
    print('Finished.')

