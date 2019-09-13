from __future__ import absolute_import, division, print_function

"""

This is a modification of the classify_images.py
script in Tensorflow. The original script produces
string labels for input images (e.g. you input a picture
of a cat and the script returns the string "cat"); this
modification reads in a directory of images and 
generates a vector representation of the image using
the penultimate layer of neural network weights.

Usage: python feature_extract.py "data_dir/*.jpg"

"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

import os.path
import re
import sys
import tarfile
import glob
import json
import psutil
import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def create_graph():
#  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(image_list, output_dir):
  """Runs inference on an image list.

  Args:
    image_list: a list of images.
    output_dir: the directory in which image vectors will be saved

  """

  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    for image_index, image in enumerate(image_list):
      try:
        print("parsing", image_index, image, "\n")
        if not tf.gfile.Exists(image):
          tf.logging.fatal('File does not exist %s', image)
        
        with tf.gfile.FastGFile(image, 'rb') as f:
          image_data =  f.read()

          # Get penultimate layer weights
          feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
          feature_set = sess.run(feature_tensor,
                          {'DecodeJpeg/contents:0': image_data})
          feature_vector = np.squeeze(feature_set)        
          outfile_name = os.path.basename(image) + ".npz"
          out_path = os.path.join(output_dir, outfile_name)
          np.savetxt(out_path, feature_vector, delimiter=',')

        # close the open file handlers
        proc = psutil.Process()
        open_files = proc.open_files()

        for open_file in open_files:
          file_handler = getattr(open_file, "fd")
          os.close(file_handler)
      except:
        print('could not process image index',image_index,'image', image)
  return


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  if len(sys.argv) < 2:
    print("please provide a glob path to one or more images, e.g.")
    print("python feature_extract.py '.data/*.jpg'")
    sys.exit()

  else:
    output_dir = "feature_vectors/"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    images = glob.glob(sys.argv[1])
    print(images)
    run_inference_on_images(images, output_dir)

    print("all done")
if __name__ == '__main__':
  tf.app.run()
