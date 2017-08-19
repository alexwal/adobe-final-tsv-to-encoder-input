# 6m_app.py

import logging
import random
import time
import os
import queue
import threading
import progressbar

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

import gensim
from gensim.models import KeyedVectors

from flask import Flask, jsonify, url_for, request
import flask

import numpy as np

import sys
# sys.path.append('/home/ubuntu/workspace/models/slim')
# sys.path.append('misc')

# Main slim library
from tensorflow.contrib import slim
import tensorflow as tf
from nets import inception
from preprocessing import inception_preprocessing

# Custom
import runner
import utils
import embeddings_6m_dataset as dataset_6m
import load_6m_dataset
import s3_utils

FLAGS = tf.app.flags.FLAGS


##### Flask setup


app = Flask(__name__, static_folder='6m-data')
app.config.from_object(__name__)


#####


# Query settings
MAX_RESULTS = 100
results = {}


# Data settings
batch_size = FLAGS.batch_size
dataset_dir = FLAGS.embedding_tfrecords_dir
TMP_IMG_DIR = '6m-data/tmp_images'


# Get int -> string label mapping
labels_file = os.path.join(FLAGS.image_tfrecords_dir, 'labels.txt')
label_to_name = {int(label) : str_label
                     for label, str_label in
                         map(lambda x : x.strip().split(':'),
                             open(labels_file, 'r')
                )}


# Create dictionary from class label to label embedding (word2vec):
stored_embeddings = np.load(FLAGS.word_embeddings_dir)[()]
classes = list(label_to_name.values())
label_embeddings = {label : utils.normalize(stored_embeddings[label]) for label in classes}


# Put me back!
# Loading embedding model
print('Loading word embedding model (word2vec)...')
embedding_model = KeyedVectors.load_word2vec_format(FLAGS.embedding_model_home, binary=True)


#####


# Helper functions

def load_6m_data():
  # Load all embeddings and paths on which to run search queries
  dataset_dir = '6m-embedding-batches'

  all_paths, all_embeddings, all_image_ids = [], [], []

  with tf.Graph().as_default():

    dataset = dataset_6m.get_dataset(dataset_dir)
    embs, labels, filenames = load_6m_dataset.load_batch(dataset, batch_size=batch_size,  num_epochs=1)
    sv = tf.train.Supervisor(logdir=None, summary_op=None)

    with sv.managed_session('') as sess:
      i = 0
      while not sv.should_stop():
        np_embs, np_labels, np_filenames = sess.run([embs, labels, filenames])
        all_paths.extend(filename for filename in np_filenames)
        all_embeddings.extend(emb for emb in np_embs)
        all_image_ids.extend(label for label in np_labels)
        if i % 1000 == 0:
          print('Loaded %d batches...' % i)
          sys.stdout.write('\rLoaded %d batches...' % i)
          sys.stdout.flush()
        i += 1

  tf.reset_default_graph()
  return all_paths, all_embeddings, all_image_ids


def get_embedding_from_word(word):
    if word not in label_embeddings:
        # Load model if necessary to compute new word embedding
        w2v = embedding_model[word]
        embedding = utils.normalize(w2v)
        return embedding
    return label_embeddings[word]


def get_embedding_from_probs(probs, T=10):
    # First, to get probs: input an image through a classifier.
    # Returns the conse vector (semantic embedding)
    # probs[i] is probability that input image belongs to class label_to_name[i].
    top_inds, top_probs = utils.top_values(probs, k=T)
    top_labels = [label_to_name[i] for i in top_inds]
    label_weights = utils.softmax(top_probs) # normalize the top T probabilities
    result = np.sum(
      [label_weights[i] * label_embeddings[top_labels[i]]
               for i in range(len(top_labels))
      ], axis=0)
    return utils.normalize(result)


def get_raw_image(url_or_path):
    # Check if input is URL or local path
    path = True if os.path.isfile(url_or_path) else False
    try:
        if path:
            raw_image = open(url_or_path, 'rb').read()
        else:
            raw_image = urllib.urlopen(url_or_path).read()
        return raw_image
    except:
        return None


def get_embedding_from_image(url_or_path):
    # Load the raw image bytes from a URL or path

    raw_image = get_raw_image(url_or_path)
    
    if raw_image is None:
        return None
    
    # Obtain base model transfer value and classifier probabilities
    with image_sv.managed_session(master='', config=config) as image_sess:
        np_image, probs = image_sess.run([decode_image, probabilities], {Raw_Image : raw_image})
    
    # Compute visual-semantic embedding of the input image using probabilities
    embedding = get_embedding_from_probs(probs)
    return embedding


def nearest_embeddings_dict(embedding, paths, embeddings):
    # Return the `batch` of embeddings and original image paths by similarity to input embedding.
    embedding = utils.normalize(embedding)
    similarity = {}
    bar = progressbar.ProgressBar(max_value=len(paths))
    i = 0
    for label, v in zip(paths, embeddings):
        bar.update(i)
        label = label.decode('utf-8')
        similarity[label] = np.dot(embedding, utils.normalize(v)) # cosine similarity (both vects are normed)
        i += 1
    bar.finish()
    return similarity


def update_results(data, results, max_results):
    # Update dictionary results with new dict data in form: {key=path : value=similarity, ...},
    # ensuring that len(results) < max_results.
    
    # Add all new data to results
    results.update(data)

    # Delete least similar data in excess of max_results
    delete_keys = sorted(results, key=results.get, reverse=True)[max_results:]
    for key in delete_keys:
        del results[key]


#### Get search dataset


all_paths, all_embeddings, all_image_ids = load_6m_data()
_NUM_CLASSES = 267 # from the VSO-dataset

#### Launch image pipeline:


# Converts a query IMAGE to classifier PROBABILITIES


image_graph = tf.Graph()

# TODO: don't hard code these:
# Base model:
checkpoint_file1 = os.path.join(FLAGS.base_model_dir, 'inception_v4.ckpt')
# Classifier model:
clsfr_dir ='data/logs/tmp' # CHANGE to FLAG
checkpoint_file2 = tf.train.latest_checkpoint(clsfr_dir)

with image_graph.as_default():


    # Decode and preprocess raw image bytes (using a placeholder for the raw image)
    Raw_Image = tf.placeholder(dtype=tf.string, shape=[])
    decode_image = tf.image.decode_image(Raw_Image, channels=3)
    decode_image.set_shape([None, None, 3])
    processed_image = inception_preprocessing.preprocess_image(decode_image, FLAGS.image_size, FLAGS.image_size, is_training=False)
    processed_image = tf.cast(tf.expand_dims(processed_image, axis=0), tf.float32)


    # Get CNN and its outputs
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        logits, endpoints = inception.inception_v4(processed_image, num_classes=1001, is_training=False)
    transfer_value_op = endpoints['PreLogitsFlatten']


    # Restore CNN variables
    variables_to_restore1 = slim.get_variables_to_restore()
    saver1 = tf.train.Saver(variables_to_restore1)


    # Classifier
    logits = slim.fully_connected(transfer_value_op, _NUM_CLASSES, activation_fn=None, scope='FC/fc1')


    # Restore trained classifier variables
    variables_to_restore2 = list(set(slim.get_variables_to_restore()).difference(variables_to_restore1))
    saver2 = tf.train.Saver(variables_to_restore2)


    # Function to run
    def restore_fn(sess):
        return saver1.restore(sess, checkpoint_file1), saver2.restore(sess, checkpoint_file2)


    # Get probabilities of each class
    probabilities = tf.nn.softmax(logits)
    probabilities = tf.squeeze(probabilities, axis=0)


# Create a Supervisor that will initialize variables and start enqueue threads.


config = tf.ConfigProto(intra_op_parallelism_threads=8, allow_soft_placement=True)
image_sv = tf.train.Supervisor(graph=image_graph, logdir=None, summary_op=None, init_fn=restore_fn)


########


# NOTE: Prepare threads OUTSIDE OF: if __name__ == '__main__'
# so that they are in global scope.


# Threading settings
num_threads = 16
# batch_size = 4
queue_size = batch_size * 8
q = queue.Queue(maxsize=queue_size)


# Threads start and wait for items to be enqueued to q
threads = []
for i in range(num_threads):
  t = s3_utils.Downloader(q, dest_dir=TMP_IMG_DIR)
  t.start() # calls run
  threads.append(t)


def batch_enqueue_paths(paths):
  # Delete previous results
  if tf.gfile.Exists(TMP_IMG_DIR):
    tf.gfile.DeleteRecursively(TMP_IMG_DIR)
  tf.gfile.MakeDirs(TMP_IMG_DIR)

  # Download new images
  for s3_path in paths:
    q.put(s3_path)
  q.join() # wait for all task_dones




########



####


# app


# Lets the server serve images to the client
@app.route('/<path:filename>')
def download_file(filename):
    return flask.send_from_directory(app.static_folder, filename)


# Word query
@app.route('/word_query')
def word_query():
    start = time.time()

    query = request.args['query']
    if not query or query == '':
        return None


    app.logger.info('Running query on word: %s' % (query))


    # TEMPORARY HACK to allow urls and local paths.
    if '/' in query:
      print('Detected IMAGE query.')
      embedding = get_embedding_from_image(query)
    else:
      print('Detected WORD query.')
      embedding = get_embedding_from_word(query)


    embeddings_dict = nearest_embeddings_dict(embedding, all_paths, all_embeddings)
    update_results(embeddings_dict, results, MAX_RESULTS)
    
    sorted_results = sorted(results.keys(), key=results.get, reverse=True)
    batch_enqueue_paths(sorted_results) # downloads images
    sorted_results = [s3_utils.format_filename(f, TMP_IMG_DIR) for f in sorted_results] # convert from s3 to local path
    
    dt = time.time() - start
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    
    return jsonify(sorted_results)


# Image query
@app.route('/image_query')
def image_query():
    start = time.time()

    query = request.args['query']
    if not query or query == '':
        return None

    app.logger.info('Running query on image: %s' % (query))

    embedding = get_embedding_from_image(query)
    if embedding is None:
        print('Embedding None. Could not retrieve image at %s' % query)
        return None
    
    embeddings_dict = nearest_embeddings_dict(embedding, all_paths, all_embeddings)
    update_results(embeddings_dict, results, MAX_RESULTS)
    
    sorted_results = sorted(results.keys(), key=results.get, reverse=True)
    batch_enqueue_paths(sorted_results) # downloads images
    sorted_results = [s3_utils.format_filename(f, TMP_IMG_DIR) for f in sorted_results] # convert from s3 to local path
    
    dt = time.time() - start
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    
    return jsonify(sorted_results)

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True, port=8080)
    # To run this app:
    # export FLASK_APP=6m_app.py
    # python -m flask run --host=0.0.0.0 --port=8080

