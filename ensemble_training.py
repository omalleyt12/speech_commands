"""Never mind, I'm just going to go through and have each model restored and save its training probabilities, then create a single-layer, L1-penalized model to find the bestest combo"""
import tensorflow as tf
print(dir(tf.contrib))
import numpy as np
import preprocessing as pp
from features import make_features
from my_models import make_model

sample_rate = 16000

models = [
    ('log-mel','overdrive','overdrive_frame_eq'),
    ('log-mel','overdrive','overdrive_long_pl1')
]

sessions = []
for feats,mod,loc in models:
    g = tf.Graph()
    with g.as_default() as cur_g:
        saver = tf.train.import_meta_graph('models/{}.ckpt.meta'.format(loc))
        sess = tf.Session(graph=cur_g)
        saver.restore(sess,'models/{}.ckpt'.format(loc))
    sessions.append(sess)
