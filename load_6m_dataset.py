import numpy as np

# Main slim library
from tensorflow.contrib import slim
import tensorflow as tf
from nets import inception
from preprocessing import inception_preprocessing

# Custom
from FINAL import utils
import embeddings_6m_dataset as dataset_6m


def load_batch(dataset, batch_size=8, num_epochs=None):
    """Loads a single batch of EMBEDDINGS.
    
    Args:
      : The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      transfer_values: A Tensor of size [batch_size, ...], ... samples that have been preprocessed.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
      filenames: A Tensor of size [batch_size], whose values...
    """
    
    # Decode TFRecords
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=batch_size * 2, #change?
        common_queue_min=0, num_epochs=num_epochs)
    embedding, label, filename = data_provider.get(['embedding', 'label', 'filename'])

    # Batch it up.
    embeddings, labels, filenames = tf.train.batch(
          [embedding, label, filename],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size + 4,
          allow_smaller_final_batch=True,
        )

    return embeddings, labels, filenames

if __name__ == '__main__':
  samples = []

  with tf.Graph().as_default():

    dataset_dir = '200K-embedding-batches'
    dataset = dataset_6m.get_dataset(dataset_dir)
    embs, labels, filenames = load_batch(dataset, batch_size=1, num_epochs=1)
    sv = tf.train.Supervisor(logdir=None, summary_op=None)

    with sv.managed_session('') as sess:
      i = 0
      while not sv.should_stop():
        np_embs, np_labels, np_filenames = sess.run([embs, labels, filenames])
        samples.extend(label for label in np_labels)
        if i % 10 == 0:
          print('Loaded %d batches...' % i)
        if i > 100:
          break
        i += 1

  print(samples, len(samples), 'num samples')
  assert len(set(samples)) == len(samples), 'Mismatch in samples obtained'

