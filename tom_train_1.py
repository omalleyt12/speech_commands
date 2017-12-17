wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
all_words = wanted_words + ["bed","bird","cat","dog","eight","five","four","happy","house","marvin","nine","one","seven","sheila","six","three","tree","two","wow","zero","true_unknown"]
all_words_index = {w:i for i,w in enumerate(all_words)}


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
    print(a.shape)
    sciwav.write("testing.wav",a.shape[0],a)
    winsound.PlaySound("testing.wav",winsouns.SND_FILENAME)

style = "unknown"
batch_size = 100
eval_step = 500
steps = 20000
learning_rate = 0.001
decay_every = 2000
decay_rate = 0.80
sample_rate = 16000 # per sec
silence_percentage = 10.0 # what percent of training data should be silence
unknown_percentage = 10.0 # what percent of training data should be unknown words
true_unknown_percentage = 10.0 # what percent of words should be complete goobledyguk


sess = tf.InteractiveSession()

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
        set_name = utils.which_set(wav_path)
        d = {"file":wav_path,"data":wav_loader.load(wav_path,sess)}
        if style == "full" or word in wanted_words:
            d.update({"label":word,"label_index":all_words_index[word]})
            data_index[set_name].append(d)
        else:
            d.update({"label":"unknown","label_index":1})
        if word not in wanted_words: # this will be populated for both styles
            unknown_index[set_name].append(d)

    # randomize each set
    for set_name in ['val','test','train']:
        random.shuffle(data_index[set_name])

    return data_index, unknown_index

def get_unknowns_by_speaker(unknown_index):
    """Organize unknowns into speakers who have spoken two or more different words"""
    def unique_words(l):
        new_l = []
        should_add = True
        for ele in l:
            for new_ele in new_l:
                if new_ele["label"] == ele["label"]:
                     should_add = False
            if should_add:
                new_l.append(ele)
        return new_l

    from collections import defaultdict
    unknown_speakers = {}
    for set_name in ['val','test','train']:
        unknown_speakers[set_name] = defaultdict(list)
        for i,rec in enumerate(unknown_index[set_name]):
            speaker = re.sub(r'_nohash_.*$','',os.path.basename(rec["file"]))
            unknown_speakers[set_name][speaker].append(rec)
        unknown_speakers[set_name] = unknown_speakers[set_name].items( )
        unknown_speakers[set_name] = [(u[0],unique_words(u[1])) for u in unknown_speakers[set_name]]
        unknown_speakers[set_name] = [u for u in unknown_speakers[set_name] if len(u[1]) > 2]
    return unknown_speakers

wav_ph = tf.placeholder(tf.float32,[None,sample_rate])

def get_batch(data_index,batch_size,offset=0,mode="train",style="full"):
    """Does preprocessing of WAVs and returns the feed_dict for a batch"""
    recs = []
    labels = []
    if offset + batch_size > len(data_index):
        batch_size = len(data_index) - offset

    for j in range(batch_size):
        samp_index = np.random.randint(len(data_index)) if mode == "train" else offset
        offset += 1 # only matters if mode != "train"

        rec = data_index[samp_index]
        rec_data = rec["data"]

        if mode == "train":
            rec_data = pp.pad(rec_data)
            rec_data = pp.add_noise(rec_data,bg_data)

        if mode != "comp":
            labels.append(rec["label_index"])
        recs.append(rec_data)

    if mode != "comp": # add silence and unknowns to batches randomly
        silence_recs = int(batch_size * silence_percentage / 100)
        for j in range(silence_recs):
            silence_rec = pp.add_noise(np.zeros(sample_rate,dtype=np.float32),bg_data)
            recs.append(silence_rec)
            labels.append(all_words_index["silence"])

        if style == "full":
            true_unknown_recs = int(batch_size * true_unknown_percentage / 100)
            for j in range(true_unknown_recs):
                speaker_words = 0
                rand_speaker = np.random.randint(0,len(unknown_speakers[mode]))
                speaker_words = unknown_speakers[mode][rand_speaker][1]
                choose_words = np.random.choice(len(speaker_words),2)
                tu_word1 = speaker_words[choose_words[0]]["data"]
                tu_word2 = speaker_words[choose_words[1]]["data"]
                scrambler_decider = np.random.uniform()
                if scrambler_decider < 0.2:
                     tu_rec = pp.reverse(tu_word1)
                elif scrambler_decider < 0.7:
                     tu_rec = pp.combine(tu_word1,tu_word2)
                else:
                     tu_rec = pp.add(tu_word1,tu_word2)
                recs.append(tu_rec)
                labels.append(len(all_words) -1) # this "true_unknown"
        else:
            unknown_recs = int(batch_size * unknown_percentage / 100)
            for j in range(unknown_recs):
                rand_unknown = np.random.randint(0,len(unknown_index[mode]))
                u_rec = unknown_index[mode][rand_unknown]["data"]
                recs.append(u_rec)
                labels.append(1)

    feed_dict={ wav_ph: np.stack(recs)}
    if mode != "comp":
        feed_dict[labels_ph] = np.stack(labels).astype(np.int32)
    return feed_dict


data_index, unknown_index = load_train_data(style=style)
unknown_speakers = get_unknowns_by_speaker(unknown_index)
bg_data = wl.load_bg_data(sess)

labels_ph = tf.placeholder(tf.int32,(None))
wav_ph = tf.placeholder(tf.float32,(None,sample_rate))
keep_prob = tf.placeholder(tf.float32) # will be 0.5 for training, 1 for test
learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")

features = make_features(wav_ph,"mfcc")

output_neurons = len(all_words) if style == "full" else len(wanted_words)
final_layer = orig_with_extra_fc(features,keep_prob,output_neurons)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph, logits=final_layer)
loss_mean = tf.reduce_mean(loss)

train_step = tf.train.AdamOptimizer(learning_rate_ph).minimize(loss_mean)
predictions = tf.argmax(final_layer,1,output_type=tf.int32)
is_correct = tf.equal(labels_ph,predictions)
confusion_matrix = tf.confusion_matrix(labels_ph,predictions,num_classes=output_neurons)
accuracy_tensor = tf.reduce_mean(tf.cast(is_correct,tf.float32))
global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step,global_step + 1)

saver = tf.train.Saver(tf.global_variables())
tf.summary.scalar("cross_entropy",loss_mean)
tf.summary.scalar("accuracy",accuracy_tensor)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train_unknown_mfcc_extrafc",sess.graph)
val_writer = tf.summary.FileWriter("logs/val_unknown_mfcc_extrafc",sess.graph)


tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())
for i in range(steps):
    if i > 0 and i % decay_every == 0:
        learning_rate = learning_rate * decay_rate
    feed_dict = get_batch(data_index["train"],batch_size,style=style)
    feed_dict.update({keep_prob: 0.5,learning_rate_ph:learning_rate})
    # now here's where we run the real, convnet part
    if i % 10 == 0:
        sum_val,acc_val,loss_val, _ = sess.run([merged_summaries,accuracy_tensor,loss_mean,train_step],feed_dict)
        train_writer.add_summary(sum_val,i)
        tf.logging.info("Step {} Accuracy {} Cross Entropy {}".format(i,acc_val,loss_val))
    else:
        sess.run(train_step,feed_dict)

    if i % eval_step == 0 or i == (steps - 1):
        set_name = "val" if not i == (steps - 1) else "test"
        val_size = len(data_index[set_name])
        val_offset = 0
        val_acc = RunningAverage()
        val_loss = RunningAverage()
        val_conf_mat = np.zeros((output_neurons,output_neurons))
        while val_offset < val_size:
            feed_dict = get_batch(data_index[set_name],batch_size,offset=val_offset,mode="val",style=style)
            feed_dict.update({keep_prob:1.0})
            val_sum_val,val_acc_val,val_loss_val,val_conf_mat_val = sess.run([merged_summaries,accuracy_tensor,loss_mean,confusion_matrix],feed_dict)
            val_writer.add_summary(val_sum_val,i)
            val_acc.add(val_acc_val)
            val_loss.add(val_loss_val)
            val_conf_mat += val_conf_mat_val
            val_offset += batch_size
        tf.logging.info("{} Step {} Val Accuracy {} Loss {}".format(set_name,i,val_acc,val_loss))
        df_words = all_words if style == "full" else wanted_words
        pd.DataFrame(val_conf_mat,columns=df_words,index=df_words).to_csv("confusion_matrix_{}.csv".format(i))

test_index = wl.load_test_data(sess)
# now here's where we run the test classification
import pandas as pd
df = pd.DataFrame([],columns=["fname","label"])
offset = 0
test_batch_size = 100
while offset < len(test_index):
    feed_dict = get_batch(test_index,test_batch_size,offset=offset,mode="comp",style=style)
    feed_dict.update({ keep_prob:1.0})
    test_pred = sess.run(predictions,feed_dict=feed_dict)
    test_labels = [all_words[test_index] for test_index in test_pred]
    for i in range(len(test_labels)):
        if test_labels[i] not in wanted_words:
            test_labels[i] = "unknown"
    test_files = [t["identifier"] for t in test_index[offset:(offset + test_batch_size)] ]
    test_batch_df = pd.DataFrame([{"fname":ti,"label":tl} for ti,tl in zip(test_files,test_labels)])
    offset += test_batch_size
    print(offset)
    df = pd.concat([df,test_batch_df])
df.to_csv("my_guesses_3.csv",index=False)

sess.close()
