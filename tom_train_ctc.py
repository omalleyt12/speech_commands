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
    '--seed',
    type=int,
    default=0
)
parser.add_argument(
    '--pseudo_labels',
    type=str,
    default=None
)
parser.add_argument(
    '--pseudo-tom',dest="pseudo_tom",action="store_true",default=False
)
parser.add_argument(
    '--no-noise',dest="noise",action="store_false",default=True
)
parser.add_argument('--super-noise',dest="super_noise",action="store_true",default=False)
parser.add_argument('--val',dest='val',action='store_true')
parser.add_argument('--no-val',dest='val',action='store_false')
parser.add_argument('--train',dest="train",action='store_true')
parser.add_argument('--no-train',dest="train",action='store_false')

FLAGS, unparsed = parser.parse_known_args()

SILENCE_TOK = ord("z") - ord("a") + 1


wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
all_words = wanted_words +["one","bed","bird","cat","dog","eight","five","four","happy","house","marvin","nine","seven","sheila","six","three","tree","two","wow","zero"]
all_words_index = {w:i for i,w in enumerate(all_words)}

import pickle
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
        set_name = utils.which_set(wav_path) if FLAGS.val else "train"
        d = {"file":wav_path,"data":wav_loader.load(wav_path,sess)}
        if style == "full" or word in wanted_words:
            d.update({"label":word,"label_index":all_words_index[word]})
            d.update({"word":word})
            d.update({"letters":[c for c in word]})
            d.update({"letter_codes":[ord(c) - ord("a") for c in word]})
            data_index[set_name].append(d)

    # randomize each set
    for set_name in ['val','test','train']:
        random.shuffle(data_index[set_name])

    return data_index, unknown_index



def get_batch(data_index,batch_size,offset=0,mode="train",style="full"):
    """Does preprocessing of WAVs and returns the feed_dict for a batch"""
    recs = []
    letters_inds = []
    letters_vals = []
    bg_wavs = []
    seq_lens = []
    if offset + batch_size > len(data_index):
        batch_size = len(data_index) - offset

    for j in range(batch_size):
        samp_index = np.random.randint(len(data_index)) if mode == "train" else offset
        offset += 1 # only matters if mode != "train"

        rec = data_index[samp_index]
        rec_data = rec["data"]

        if mode != "comp":
            letters_inds += [(j,i) for i in range(len(rec["letter_codes"]))]
            letters_vals += rec["letter_codes"]
        seq_lens.append(len(rec["letter_codes"]))
        recs.append(rec_data)
        bg_wavs.append(pp.get_noise(bg_data,noise=FLAGS.noise,super_noise=FLAGS.super_noise)) # won't add noise to regular examples if flag is set

    k = 0
    # if mode != "comp": # add silence and unknowns to batches randomly
    #     silence_recs = int(batch_size * silence_percentage / 100)
    #     for k in range(silence_recs):
    #         silence_rec = np.zeros(sample_rate,dtype=np.float32) 
    #         if mode == "val":
    #             silence_rec += pp.get_noise(bg_data,val=True,super_noise=FLAGS.super_noise) # since noise won't be added to any val data records
    #         recs.append(silence_rec)
    #         seq_lens.append(1)
    #         letters_inds.append((j+k+1,0))
    #         letters_vals.append(SILENCE_TOK)
    #         bg_wavs.append(pp.get_noise(bg_data))

    feed_dict={ wav_ph: np.stack(recs), bg_wavs_ph: np.stack(bg_wavs), use_full_layer: False, slow_down: False}
    if mode != "comp":
        let_shape = np.array([j+k+1,12])  # we'll say the max length of a word is 12
        thingy = tf.SparseTensorValue(np.array(letters_inds),np.array(letters_vals),let_shape)
        print(thingy)
        feed_dict[targets] = thingy
        feed_dict[seq_len] = np.array(seq_lens).astype(np.int32)
        print(seq_lens)
    return feed_dict

def run_validation(set_name,step):
    # update the variance on batch normalization without dropout
    # for _ in range(50):
    #     feed_dict = get_batch(data_index["train"],batch_size,style=style)
    #     feed_dict.update({keep_prob: 1.0,learning_rate_ph:learning_rate,is_training_ph: False})
    #     up_blah,loss_blah = sess.run([update_ops,loss_mean],feed_dict)
    val_size = len(data_index[set_name])
    val_offset = 0
    val_acc = RunningAverage()
    val_loss = RunningAverage()
    j = 0
    comparison = []
    while val_offset < val_size:
        feed_dict = get_batch(data_index[set_name],batch_size,offset=val_offset,mode="val",style=style)
        feed_dict.update({keep_prob:1.0,is_training_ph:False})
        val_ler, val_loss_val = sess.run([ler,cost],feed_dict)
        val_acc_val = 1 - val_ler
        val_acc.add(val_acc_val)
        val_loss.add(val_loss_val)
        val_offset += batch_size
        j += 1

    tf.logging.info("{} Step {} LR {} Letter Accuracy {} Loss {}".format(set_name,step,learning_rate,val_acc,val_loss))



    return val_loss.calculate(), val_acc.calculate()

if FLAGS.train:
    data_index, unknown_index = load_train_data(style=style)
    print("Length of unknown_index for train {}".format(len(unknown_index["train"])))
    print("Length of training data {}".format(len(data_index["train"])))
    # full_data_index, _ = load_train_data("full")
    # speakers = get_speakers(unknown_index,data_index)
bg_data = wl.load_bg_data(sess)


wav_ph = tf.placeholder(tf.float32,(None,sample_rate))
bg_wavs_ph = tf.placeholder(tf.float32,[None,sample_rate])
targets = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None])

keep_prob = tf.placeholder(tf.float32) # will be 0.5 for training, 1 for test
learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")
is_training_ph = tf.placeholder(tf.bool)
use_full_layer = tf.placeholder(tf.bool)
slow_down = tf.placeholder(tf.bool)
# scale_means_ph = tf.placeholder(tf.float32)
# scale_stds_ph = tf.placeholder(tf.float32)


processed_wavs = pp.tf_preprocess(wav_ph,bg_wavs_ph,is_training_ph,slow_down)

features = make_features(processed_wavs,is_training_ph,FLAGS.features)
num_phonemes = ord("z") - ord("a") + 2
fingerprint_4d = tf.reshape(features,[-1,features.shape[1],features.shape[2],1])
c = conv2d(fingerprint_4d,64,[7,3],is_training_ph,mp=[1,3])
c = conv2d(c,128,[1,7],is_training_ph,mp=[1,4])
c = conv2d(c,256,[1,10],is_training_ph,padding="VALID")
c = tf.reshape(c,[-1,100,256])

saver = tf.train.Saver()
# saver.restore(sess,"models/overdrive.ckpt")

num_hidden = 256
num_layers = 1
cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                    state_is_tuple=True)
# outputs, _ = tf.nn.dynamic_rnn(stack, c, seq_len, dtype=tf.float32)
outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)
outputs = tf.reshape(outputs, [-1, num_hidden])
W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_phonemes],
                                        stddev=0.1))
# Zero initialization
# Tip: Is tf.zeros_initializer the same?
b = tf.Variable(tf.constant(0., shape=[num_phonemes]))

# Doing the affine projection
logits = tf.matmul(outputs, W) + b
logits = tf.reshape(logits,[-1,100,num_phonemes])
logits = tf.transpose(logits, (1, 0, 2))




loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)

# Option 2: tf.nn.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate_ph)

train_step = optimizer.minimize(cost)




tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())



if FLAGS.train:
    last_val_loss = 9999999
    should_stop = False
    for i in range(steps):
        if FLAGS.val:
            if i > 0 and i % 500 == 0:
                learning_rate = 0.9*learning_rate
        else:
            cum_no_val_step = 0
            for no_val_i,no_val_step in enumerate(no_val_steps):
                cum_no_val_step += no_val_step
                if cum_no_val_step > i:
                    learning_rate = no_val_lr[no_val_i]
                    break

        feed_dict = get_batch(data_index["train"],batch_size,style=style)
        feed_dict.update({keep_prob: train_keep_prob,learning_rate_ph:learning_rate,is_training_ph: True})
        # now here's where we run the real, convnet part
        if i % 10 == 0:
            _, acc_val,loss_val, _ = sess.run([update_ops,ler,cost,train_step],feed_dict)
            acc_val = 1 - acc_val
            tf.logging.info("Step {} LR {} Accuracy {} Cross Entropy {}".format(i,learning_rate,acc_val,loss_val))

            # full_feed_dict = get_batch(full_data_index["train"],batch_size,style="full")
            # full_feed_dict.update({keep_prob: train_keep_prob,learning_rate_ph:learning_rate,is_training_ph:True,use_full_layer:True})
            # _, sum_val,acc_val,loss_val, _ = sess.run([update_ops,merged_summaries,accuracy_tensor,loss_mean,train_step],full_feed_dict)
            # tf.logging.info("Full Step {} LR {} Accuracy {} Cross Entropy {}".format(i,learning_rate,acc_val,loss_val))
        else:
            sess.run([update_ops,train_step],feed_dict)

        if FLAGS.val:
            if i % eval_step == 0 or i == (steps - 1):
                val_loss, val_acc = run_validation("val",i)
                if val_loss > last_val_loss:
                    learning_rate = 0.5*learning_rate
                    print("CHANGING LEARNING RATE TO: {}".format(learning_rate))
                    # print("Restoring former model and rerunning validation")
                    # saver.restore(sess,"models/{}.ckpt".format(FLAGS.save))
                    # val_acc = run_validation("val",i)
                # else:
                #     saver.save(sess,"./model.ckpt")

                last_val_loss = val_loss

        if learning_rate < 0.00001: # at this point, just stop
            break
    saver.save(sess,"models/{}.ckpt".format(FLAGS.save))
    if FLAGS.val:
        test_loss, test_acc = run_validation("test",i)
        MAVs, MR_MODELS = run_validation("train",i)
    del data_index # need to conserve RAM

if not FLAGS.train:
    saver.restore(sess,"models/{}.ckpt".format(FLAGS.save))

test_index = wl.load_test_data(sess)
# now here's where we run the test classification
import pandas as pd
df = pd.DataFrame([],columns=["fname","label"])
offset = 0
test_batch_size = batch_size
test_info = []
while offset < len(test_index):
    if offset % 1000 == 0: print(str(offset))
    # print(offset)
    feed_dict = get_batch(test_index,test_batch_size,offset=offset,mode="comp",style=style)
    feed_dict.update({ keep_prob:1.0,is_training_ph:False})
    test_lets = sess.run([decoded],feed_dict=feed_dict)
    
    test_words = []
    test_files = [t["identifier"] for t in test_index[offset:max(offset + test_batch_size,len(test_index))] ]

    test_labels = [all_words[test_index] for test_index in test_pred]
    for i in range(len(test_labels)):
        if test_labels[i] not in wanted_words:
            test_labels[i] = "unknown"
    test_batch_df = pd.DataFrame([{"fname":ti,"label":tl} for ti,tl in zip(test_files,test_labels)])
    offset += test_batch_size
    df = pd.concat([df,test_batch_df])

df.to_csv("models/{}_guesses.csv".format(FLAGS.save),index=False)
sess.close()
with open("models/{}_comp.pickle".format(FLAGS.save),"wb") as f:
    pickle.dump(test_info,f)

from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "AC7ef9b2470e5800d2cf47640564e18f3f"
# Your Auth Token from twilio.com/console
auth_token  = "e36bb44f5528a0bcbe45509b113d9469"

if FLAGS.val:
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        to="+19082682005",
        from_="+12673607895",
        body="Model finished running with {} val accuracy and {} val loss".format(val_acc,val_loss)) 
else:
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        to="+19082682005",
        from_="+12673607895",
        body="Model finished running")


print(message.sid)

