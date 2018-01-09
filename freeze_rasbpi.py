import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
   '--features',
    type=str,
    default='blah')
parser.add_argument(
    '--model',
    type=str,
    default='blah'
)
parser.add_argument(
    '--save',
    type=str,
    default='blah'
)
parser.add_argument(
    '--freeze',
    type=str,
    default=None
)


FLAGS, unparsed = parser.parse_known_args()

save_to = "{}.pb".format(FLAGS.freeze)



import numpy as np
import tensorflow as tf
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util
import preprocessing as pp
import features as ft
import my_models as models
from tensorflow.python.framework import graph_util

keep_prob = tf.constant(1.0,dtype=tf.float32)
is_training_ph = tf.constant(False)
use_full_layer = tf.constant(False)
slow_down = tf.constant(False)
bg_wavs_ph = tf.constant(np.zeros((1,16000),np.float32))

wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
# decoded_sample_data = contrib_audio.decode_wav(
#       wav_data_placeholder,
#       desired_channels=1,
#       desired_samples=16000,
#       name='decoded_sample_data')

wavs_ph, sr = tf.py_func(lambda x: (np.zeros((1,16000),np.float32),np.array([16000],np.float32)),[],Tout=[tf.float32,tf.float32],name="decoded_sample_data")
print(wavs_ph.name)
print(sr.name)

wavs_ph = tf.reshape(wavs_ph,[1,16000])
# final_layer = tf.contrib.layers.fully_connected(wavs_ph,12)
processed_wavs = pp.tf_preprocess(wavs_ph,bg_wavs_ph,is_training_ph,slow_down)
features = ft.make_features(processed_wavs,is_training_ph,FLAGS.features)

final_layer, _, _ = models.make_model(FLAGS.model,features,keep_prob,12,32,is_training_ph)

tf.nn.softmax(final_layer,name="labels_softmax")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess,"models/{}.ckpt".format(FLAGS.save))

frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['labels_softmax'])
tf.train.write_graph(
    frozen_graph_def,
    "rasbpi_models",
    save_to,
    as_text=False)
tf.logging.info('Saved frozen graph to %s', save_to)
