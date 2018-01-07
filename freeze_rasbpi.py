import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util
import preprocessing as pp
import features as ft
import my_models as models

keep_prob = tf.constant(1.0,dtype=tf.float32)
is_training_ph = tf.constant(False)
use_full_layer = tf.constant(False)
slow_down = tf.constant(False)
bg_wavs_ph = tf.constant(np.zeros((1,16000),np.float32))

wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=16000,
      name='decoded_sample_data')

wavs_ph = tf.reshape(decoded_sample_data.audio,[1,16000])
processed_wavs = pp.tf_preprocess(wavs_ph,bg_wavs_ph,is_training_ph,slow_down)
features = ft.make_features(processed_wavs,is_training_ph,"log-mels")

final_layer, _, _ = models.newdrive(features,keep_prob,12,30,is_training_ph)

tf.nn.softmax(final_layer,name="labels_softmax")

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,"./model.ckpt")
