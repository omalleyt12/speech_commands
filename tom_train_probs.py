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
    '--seed',
    type=int,
    default=0
)
parser.add_argument('--extreme-time',dest="extreme_time",action="store_true",default=False)
parser.add_argument('--restore',type=str,default='blah')
parser.add_argument('--pseudo_labels',type=str,default=None)

FLAGS, unparsed = parser.parse_known_args()


wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
all_words = wanted_words +["one","bed","bird","cat","dog","eight","five","four","happy","house","marvin","nine","seven","sheila","six","three","tree","two","wow","zero"]
all_words_index = {w:i for i,w in enumerate(all_words)}

import pickle
import libmr
import scipy.spatial.distance as spd
import pandas as pd
import re
import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import scipy.io.wavfile as sciwav
import preprocessing as pp
import wav_loader as wl
import utils
from features import make_features
from my_models import *
from running_average import RunningAverage

def play(a):
    import winsound
    import scipy.io.wavfile as sciwav
    print(a.shape)
    sciwav.write("testing.wav",16000,a)
    winsound.PlaySound("testing.wav",winsound.SND_FILENAME)

style = "unknown"
train_keep_prob = 0.5
batch_size = 100
eval_step = 500
steps = 2000000
no_val_steps = [1000,1000,1000,3000,3000,5000,3000,1]
no_val_lr = [0.01,0.005,0.0025,0.001,0.0005,0.0001,0.00005,1e-8]
# no_val_steps = [15,15,15]
# no_val_lr = [0.01,0.001,1e-8]
learning_rate = 0.01
# decay_every = 2000
decay_rate = 0.10
sample_rate = 16000 # per sec
silence_percentage = 10.0 # what percent of training data should be silence
unknown_percentage = 10.0 # what percent of training data should be unknown words
true_unknown_percentage = 10.0 # what percent of words should be complete goobledyguk


sess = tf.InteractiveSession()
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
from keras import backend as K
K.set_session(sess)

def load_train_data(style="full"):
    """
    Style controls whether it learns all labels and then maps unwanted to unknown ("full"), or learns the unknown label directly ("unknown").
    Theoretically, option 1 should help it learn better features, but option 2 may help it generalize better.
    """
    random.seed(111)
    data_index = {"val":[],"test":[],"train":[]}
    unknown_index = {"val":[],"test":[],"train":[]}

    wav_loader = wl.WavLoader("train_wav_loader",desired_samples=sample_rate)
    for i,wav_path in enumerate(gfile.Glob("train/audio/*/*.wav")):
        if i % 1000 == 0: print("Loading training " + str(i))
        word = os.path.split(os.path.dirname(wav_path))[-1].lower()
        if word == "_background_noise_":
            continue # don't include yet
        set_name = "train"
        d = {"file":wav_path,"data":wav_loader.load(wav_path,sess)}
        if style == "full" or word in wanted_words:
            d.update({"label":word,"label_index":all_words_index[word]})
            d.update({"word":word})
            data_index[set_name].append(d)
        else:
            d.update({"label":"unknown","label_index":1})
            d.update({"word":word})
        if word not in wanted_words: # this will be populated for both styles
            unknown_index[set_name].append(d)

    # randomize each set
    for set_name in ['val','test','train']:
        random.shuffle(data_index[set_name])

    return data_index, unknown_index






def get_batch(data_index,batch_size,offset=0,mode="train",style="full"):
    """Does preprocessing of WAVs and returns the feed_dict for a batch"""
    recs = []
    labels = []
    bg_wavs = []
    if offset + batch_size > len(data_index):
        batch_size = len(data_index) - offset

    for j in range(batch_size):
        samp_index = np.random.randint(len(data_index)) if mode == "train" else offset
        offset += 1 # only matters if mode != "train"

        rec = data_index[samp_index]
        rec_data = rec["data"]

        if mode != "comp":
            labels.append(rec["label_index"])
        recs.append(rec_data)
        bg_wavs.append(pp.get_noise(bg_data)) # won't add noise to regular examples if flag is set

    if mode != "comp": # add silence and unknowns to batches randomly
        silence_recs = int(batch_size * silence_percentage / 100)
        for j in range(silence_recs):
            pseudo_picker = 0 if mode == "val" or FLAGS.pseudo_labels is None else np.random.uniform(0,1)
            if pseudo_picker < 0.5:
                silence_rec = np.zeros(sample_rate,dtype=np.float32) 
                if mode == "val":
                    silence_rec += pp.get_noise(bg_data,val=True) # since noise won't be added to any val data records
                recs.append(silence_rec)
            else:
                recs.append(pseudo_silence[np.random.randint(0,len(pseudo_silence)-1)])

            labels.append(all_words_index["silence"])
            bg_wavs.append(pp.get_noise(bg_data))

        if style != "full":
            unknown_recs = int(batch_size * unknown_percentage / 100)
            for j in range(unknown_recs):
                pseudo_picker = 0 if mode == "val" or FLAGS.pseudo_labels is None else np.random.uniform(0,1)
                if pseudo_picker < 0.5:
                    # if mode == "val": # Try and make val more deterministic
                    #     rand_unknown = offset + j
                    # else:
                    rand_unknown = np.random.randint(0,len(unknown_index["train"]))
                    u_rec = unknown_index["train"][rand_unknown]["data"]
                    recs.append(u_rec)
                else:
                    recs.append(pseudo_unknown[np.random.randint(0,len(pseudo_unknown)-1)])
                labels.append(1)
                bg_wavs.append(pp.get_noise(bg_data)) # won't add noise to unknowns is flag is set

    feed_dict={ wav_ph: np.stack(recs), bg_wavs_ph: np.stack(bg_wavs), use_full_layer: False, slow_down: False}
    if mode != "comp":
        feed_dict[labels_ph] = np.stack(labels).astype(np.int32)
    return feed_dict

def run_validation(set_name,step):
    if set_name == "train":
        from collections import defaultdict
        AVs = defaultdict(list)
    val_size = len(data_index[set_name])
    val_offset = 0
    val_acc = RunningAverage()
    val_loss = RunningAverage()
    val_conf_mat = np.zeros((output_neurons,output_neurons))
    pred_df_list = []
    errors = []
    got_ems = []
    model_probs = []
    while val_offset < val_size:
        feed_dict = get_batch(data_index[set_name],batch_size,offset=val_offset,mode="val",style=style)
        feed_dict.update({keep_prob:1.0,is_training_ph:False})
        val_probs,open_max, val_correct, val_pred,val_acc_val,val_loss_val,val_conf_mat_val = sess.run([probabilities,open_max_layer,is_correct,predictions,accuracy_tensor,loss_mean,confusion_matrix],feed_dict)
        val_acc.add(val_acc_val)
        val_loss.add(val_loss_val)
        val_conf_mat += val_conf_mat_val
        val_pred_list = list(val_pred)

        for i,guess in enumerate(list(val_correct)):
            if not guess:
                errors.append({
                    "data":feed_dict[wav_ph][i],
                    "guess":val_pred[i],
                    "label":feed_dict[labels_ph][i]
                })
            else:
                got_ems.append({
                    "data":feed_dict[wav_ph][i],
                    "guess":val_pred[i],
                    "label":feed_dict[labels_ph][i]
                })
            # save the probabilities to be used in creating an ensemble model
            model_probs.append({
                "label":feed_dict[labels_ph][i],
                "all_prob":val_probs[i]
            })
        val_offset += batch_size


    with open("errors_{}".format(set_name),"wb") as f:
        pickle.dump(errors,f)
    with open("got_ems_{}".format(set_name),"wb") as f:
        pickle.dump(got_ems,f)
    with open("train_probs/{}.csv".format(FLAGS.restore),"wb") as f:
        pickle.dump(model_probs,f)


    tf.logging.info("{} Step {} LR {} Accuracy {} Loss {}".format(set_name,step,learning_rate,val_acc,val_loss))
    df_words = all_words if style == "full" else wanted_words

    pd.DataFrame(val_conf_mat,columns=df_words,index=df_words).to_csv("confusion_matrix_{}.csv".format(set_name))


    return val_loss.calculate(), val_acc.calculate()


data_index, unknown_index = load_train_data(style=style)
print("Length of unknown_index for train {}".format(len(unknown_index["train"])))
print("Length of training data {}".format(len(data_index["train"])))
bg_data = wl.load_bg_data(sess)


labels_ph = tf.placeholder(tf.int32,(None))
wav_ph = tf.placeholder(tf.float32,(None,sample_rate))
bg_wavs_ph = tf.placeholder(tf.float32,[None,sample_rate])

keep_prob = tf.placeholder(tf.float32) # will be 0.5 for training, 1 for test
learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")
is_training_ph = tf.placeholder(tf.bool)
use_full_layer = tf.placeholder(tf.bool)
slow_down = tf.placeholder(tf.bool)
# scale_means_ph = tf.placeholder(tf.float32)
# scale_stds_ph = tf.placeholder(tf.float32)

processed_wavs = pp.tf_preprocess(wav_ph,bg_wavs_ph,is_training_ph,slow_down,extreme=FLAGS.extreme_time)

features = make_features(processed_wavs,is_training_ph,FLAGS.features)

output_neurons = len(all_words) if style == "full" else len(wanted_words)
full_output_neurons = len(all_words)
final_layer, full_final_layer, open_max_layer = make_model(FLAGS.model,features,keep_prob,output_neurons,full_output_neurons,is_training_ph)

final_layer = tf.cond(use_full_layer,lambda: full_final_layer, lambda: final_layer)

probabilities = tf.nn.softmax(final_layer)

loss_mean = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph, logits=final_layer)
# full_loss_mean = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph,logits=full_final_layer)

total_loss = tf.losses.get_total_loss()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate_ph)

train_step = optimizer.minimize(total_loss)
# full_train_step = optimizer.minimize(full_loss_mean)

predictions = tf.argmax(final_layer,1,output_type=tf.int32)
is_correct = tf.equal(labels_ph,predictions)
confusion_matrix = tf.confusion_matrix(labels_ph,predictions,num_classes=output_neurons)
accuracy_tensor = tf.reduce_mean(tf.cast(is_correct,tf.float32))
global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step,global_step + 1)

tf.summary.scalar("cross_entropy",loss_mean)
tf.summary.scalar("accuracy",accuracy_tensor)


tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

saver.restore(sess,"models/{}.ckpt".format(FLAGS.restore))
print("Restored it")

run_validation("train",0)
