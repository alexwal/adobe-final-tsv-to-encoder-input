import tensorflow as tf
import os
import re
import progressbar

# Make sure these are all up-to-date.
import boto3
import boto3.session
import botocore
import threading
import queue



class Downloader(threading.Thread):
  '''
    Every Downloader object is thread that
    downloads a dequeues s3_path.
  '''

  def __init__(self, queue, dest_dir='image-batches'):
      threading.Thread.__init__(self)
      self.q = queue
      self.dest_dir = dest_dir


  def run(self):
      session = boto3.session.Session()
      s3 = session.resource('s3')
      # ... do some work with S3 ...
      while True:
          item = self.q.get()
          if item is None:
            break
          self.download(item, s3)
          self.q.task_done()


  def download(self, item, s3):
      # img_id, s3_path = item (ORIGINAL, so commented code won't work if this is not here)
      s3_path = item # NEW
      if hasattr(s3_path, 'decode'):
        # convert b to str
        s3_path = s3_path.decode('utf-8')
      dest_file = format_filename(s3_path, self.dest_dir)
      download_s3_path(s3_path, dest_file, s3)

#####


# Helper functions for s3 downloads


def download_s3_path(path, dest_file, s3):
  bucket_name, key = s3_path_to_bucket_location(path)
  bucket = s3.Bucket(bucket_name)
  try:
      bucket.download_file(key, dest_file)
  except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == '404':
          print('The object does not exist: ', key)
      else:
          print('Error getting key: ', key, 'from bucket:', bucket_name)
          raise


def format_filename(path, dest_dir):
  # Format dest file of path on s3 to place in local dest dir
  if hasattr(path, 'decode'):
    # convert b to str
    path = path.decode('utf-8')
  return os.path.join(dest_dir, os.path.basename(path))


def s3_path_to_bucket_location(path):
  # Sample path: 's3://adobe-stock-search-uw2/images/jpg/00/10/00/00/1000_F_10000068_pfnm7NIEI4Ag7B83ifOesQeMLpBxT7x8_NW.jpg'
  pattern = 's3://([^/]*)/(.*.jpg)'
  bucket_name, key = re.findall(pattern, path)[0]
  return bucket_name, key

'''

########


# NOTE: Prepare threads OUTSIDE OF: if __name__ == '__main__'
# so that they are in global scope.


# Threading settings
num_threads = 16
batch_size = 4
queue_size = batch_size * 8
q = queue.Queue(maxsize=queue_size)


# Threads start and wait for items to be enqueued to q
threads = []
for i in range(num_threads):
  t = Downloader(q)
  t.start() # calls run
  threads.append(t)

def batch_enqueue_paths(ids, paths):
  for img_id, s3_path in zip(ids, paths):
    q.put((img_id, s3_path))

########


if __name__ == '__main__':


  INPUT_CSV = 'excerpt.tsv'
  # batch_size = 4
  bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)


  with tf.device('/cpu:0'):


    # Decode row of CSV (Each row: ID \t S3_FILE_PATH)
    filename_queue = tf.train.string_input_producer([INPUT_CSV], num_epochs=1)
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(records=value, record_defaults=[[''], ['']], field_delim='\t')


    # Enqueue a batch of rows
    image_id, image_s3_path = tf.train.batch(decoded,
                                            batch_size=batch_size,
                                            num_threads=1,
                                            capacity=2 * batch_size + 4,
                                            allow_smaller_final_batch=True)


  # Create a Supervisor that will initialize variables and start enqueue threads that read from TSV file.
  sv = tf.train.Supervisor(logdir=None, summary_op=None)


  # Download and process batch from the 6 million
  i = 0
  with sv.managed_session(master='') as sess:
    while not sv.should_stop():
      bar.update(i)
      i += 1

      # Get paths and store image bytes locally
      ids, paths = sess.run([image_id, image_s3_path])

      # To download and process, keep enqueueing items to q
      batch_enqueue_paths(ids, paths)


  # Wait for all threads and queue to finish
  q.join()
  for i in range(queue_size):
    q.put(None)
  for t in threads:
    t.join()


'''
