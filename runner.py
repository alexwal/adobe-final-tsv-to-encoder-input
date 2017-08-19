import tensorflow as tf

# Basic settings and model parameters.

# NEW
tf.app.flags.DEFINE_string('json_dest', 'tmp.json', 'destionation')
tf.app.flags.DEFINE_string('embeddings_dir', 'tmp-emb-batches','origin')
tf.app.flags.DEFINE_string('num_samples_dict', 'embedding_num_samples.npy','recorded sample count')

# Data paths
tf.app.flags.DEFINE_string('images_dir', 'data/images/raw',
                        'Path to image directory.')
tf.app.flags.DEFINE_string('image_tfrecords_dir', 'data/images/tfrecords',
                        'Path to image directory.')
tf.app.flags.DEFINE_string('transfer_value_tfrecords_dir', 'data/transfer_values/tfrecords',
                        'Path to image directory.')
tf.app.flags.DEFINE_string('embedding_tfrecords_dir', 'data/embeddings/tfrecords',
                        'Path to image directory.')
tf.app.flags.DEFINE_string('word_embeddings_dir', 'data/word_embeddings/word_embeddings.npy',
                        'Where to save label word embeddings (normed).')
tf.app.flags.DEFINE_string('tmp', 'data/tmp',
                        'Short term storage that is okay to delete/rewrite at any time.')
tf.app.flags.DEFINE_string('log_dir', 'data/logs',
                        'Short term storage that is okay to delete/rewrite at any time.')
# LOCAL: tf.app.flags.DEFINE_string('tfslim_dir', '/Users/walczak/workspace/models/slim', 'Where we should look for TF-Slim module')
tf.app.flags.DEFINE_string('tfslim_dir', '/home/ubuntu/workspace/models/slim', 'Where we should look for TF-Slim module') # ubuntu

# Models

# Pretrained:
tf.app.flags.DEFINE_string('base_model_dir', 'data/models/base', '...')
tf.app.flags.DEFINE_string('embedding_model_home', 'data/models/word_embedding/GoogleNews-vectors-negative300.bin',
                        'Path to the pretrained embedding model (w2v).')

# Model training and data settings
tf.app.flags.DEFINE_float('test_fraction', 0.05, 'percentage of images to put aside for test.')
tf.app.flags.DEFINE_integer('transfer_value_size', 1536,
                        'Dimension of transfer value output vector.')
tf.app.flags.DEFINE_integer('epochs', 1, '...') # how many times to cycle over data (if None, loops indefinitely)
tf.app.flags.DEFINE_integer('transfer_value_shards', 16, '...')
tf.app.flags.DEFINE_integer('embedding_shards', 4, '...')
tf.app.flags.DEFINE_integer('image_shards', 64, '...')
tf.app.flags.DEFINE_integer('image_size', 299, '...')
tf.app.flags.DEFINE_integer('embedding_size', 300, 'w2v size')
tf.app.flags.DEFINE_integer('num_threads', 4, '...')
tf.app.flags.DEFINE_integer('batch_size', 64, '...')
tf.app.flags.DEFINE_integer('topk', 4, 'TODO')
tf.app.flags.DEFINE_integer('precision', 10, 'used for evaluating embeddings and showing nearest images I think')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data before tfr etc')
tf.app.flags.DEFINE_boolean('show',           False, 'show results and image in query.')
tf.app.flags.DEFINE_boolean('continue_training', False, 'pickup training where we last left off')

# Query
# Lena: 'http://tech.velmont.net/files/2009/04/lenna-lg.jpg',
tf.app.flags.DEFINE_string('image', 'data/images/raw/wet_snow/2910297818_9a72c7a8d5.jpg',
                          'Input an image URL or path to run query')
tf.app.flags.DEFINE_string('word', 'fun',
                          'Input a word to run query on')

# Dataset
tf.app.flags.DEFINE_string('stage', 'both', 'test | train | both')

# Run stages
tf.app.flags.DEFINE_boolean('prepare' ,         False, '...')
tf.app.flags.DEFINE_boolean('conse',            False, '...')
tf.app.flags.DEFINE_boolean('convert',          False, '...')
tf.app.flags.DEFINE_boolean('images',           False, '...')
tf.app.flags.DEFINE_boolean('transfer_values',  False, '...')
tf.app.flags.DEFINE_boolean('train',            False, '...')
tf.app.flags.DEFINE_boolean('image_embeddings',           False, '...')

FLAGS = tf.app.flags.FLAGS

def main(_):

  # Check if flags meet reqs here, examine if dirs exist, delete others?
  assert FLAGS.stage in ['train', 'test', 'both'], 'Stage must be train, test, or both.'

  if FLAGS.convert:
    # [2] convert images to tfrecords
    import convert_embeddings_to_json
    convert_embeddings_to_json.run()

  if FLAGS.train:
    # [4] train
    import train_classifier
    train_classifier.run()

  if FLAGS.image_embeddings: # conse
    # [5] compute image embeddings
    import create_embedding_tfrecords
    create_embedding_tfrecords.run()

if __name__ == '__main__':
  tf.app.run()

