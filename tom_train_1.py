wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
all_words = wanted_words + ["bed","bird","cat","dog","eight","five","four","happy","house","marvin","nine","one","seven","sheila","six","three","tree","two","wow","zero","true_unknown"]
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
batch_size = 100
eval_step = 500
steps = 2000000
learning_rate = 0.01
# decay_every = 2000
decay_rate = 0.10
sample_rate = 16000 # per sec
silence_percentage = 10.0 # what percent of training data should be silence
unknown_percentage = 10.0 # what percent of training data should be unknown words
true_unknown_percentage = 10.0 # what percent of words should be complete goobledyguk

np.random.seed(0)
tf.set_random_seed(0)

sess = tf.InteractiveSession()
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
        set_name = utils.which_set(wav_path)
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

def get_speakers(unknown_index,data_index):
    """Organize unknowns into speakers who have spoken two or more different words"""
    def unique_words(l):
        new_l = []
        should_add = True
        for ele in l:
            for new_ele in new_l:
                if new_ele["word"] == ele["word"]:
                     should_add = False
            if should_add:
                new_l.append(ele)
        return new_l

    full_index = {}
    for set_name in ["train","val","test"]:
        full_index[set_name] = unknown_index[set_name] + data_index[set_name]

    from collections import defaultdict
    speakers = {}
    for set_name in ['val','test','train']:
        speakers[set_name] = defaultdict(list)
        for i,rec in enumerate(full_index[set_name]):
            speaker = re.sub(r'_nohash_.*$','',os.path.basename(rec["file"]))
            speakers[set_name][speaker].append(rec)
        speakers[set_name] = speakers[set_name].items( )
        speakers[set_name] = [(u[0],unique_words(u[1])) for u in speakers[set_name]]
        speakers[set_name] = [u for u in speakers[set_name] if len(u[1]) > 2]
    return speakers


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
        bg_wavs.append(pp.get_noise(bg_data))

    if mode != "comp": # add silence and unknowns to batches randomly
        silence_recs = int(batch_size * silence_percentage / 100)
        for j in range(silence_recs):
            silence_rec = np.zeros(sample_rate,dtype=np.float32) 
            if mode == "val":
                silence_rec += pp.get_noise(bg_data) # since noise won't be added to any val data records
            recs.append(silence_rec)
            labels.append(all_words_index["silence"])
            bg_wavs.append(pp.get_noise(bg_data))

        unknown_recs = int(batch_size * unknown_percentage / 100)
        for j in range(unknown_recs):
            if mode == "val": # Try and make val more deterministic
                rand_unknown = j
            else:
                rand_unknown = np.random.randint(0,len(unknown_index[mode]))
            u_rec = unknown_index[mode][rand_unknown]["data"]
            recs.append(u_rec)
            labels.append(1)
            bg_wavs.append(pp.get_noise(bg_data))

    feed_dict={ wav_ph: np.stack(recs), bg_wavs_ph: np.stack(bg_wavs)}
    if mode != "comp":
        feed_dict[labels_ph] = np.stack(labels).astype(np.int32)
    return feed_dict

def run_validation(set_name,step):
    # update the variance on batch normalization without dropout
    for _ in range(50):
        feed_dict = get_batch(data_index["train"],batch_size,style=style)
        feed_dict.update({keep_prob: 1.0,learning_rate_ph:learning_rate,is_training_ph: True})
         _,_ = sess.run([update_ops,loss_mean],feed_dict)
    if set_name == "train":
        from collections import defaultdict
        AVs = defaultdict(list)
    val_size = len(data_index[set_name])
    val_offset = 0
    val_acc = RunningAverage()
    val_loss = RunningAverage()
    val_conf_mat = np.zeros((output_neurons,output_neurons))
    pred_df_list = []
    while val_offset < val_size:
        feed_dict = get_batch(data_index[set_name],batch_size,offset=val_offset,mode="val",style=style)
        feed_dict.update({keep_prob:1.0,is_training_ph:False})
        open_max, val_correct, val_pred,val_sum_val,val_acc_val,val_loss_val,val_conf_mat_val = sess.run([open_max_layer,is_correct,predictions,merged_summaries,accuracy_tensor,loss_mean,confusion_matrix],feed_dict)
        val_writer.add_summary(val_sum_val,step)
        val_acc.add(val_acc_val)
        val_loss.add(val_loss_val)
        val_conf_mat += val_conf_mat_val
        val_offset += batch_size
        val_pred_list = list(val_pred)
        for val_rec,val_p in zip(data_index[set_name][val_offset:val_offset+batch_size],val_pred_list[:batch_size]):
            pred_df_list.append({
                "true_label":val_rec["label"],
                "true_word":val_rec["word"],
                "pred_label":all_words[val_p],
                "file":val_rec["file"]
            })
        silence_end = batch_size + int(batch_size * silence_percentage / 100)
        for val_p in val_pred_list[batch_size:silence_end]:
            pred_df_list.append({
                "true_label":"silence",
                "true_word":"silence",
                "pred_label":all_words[val_p],
                "file":None
            })
        for val_p in val_pred_list[silence_end:]:
            pred_df_list.append({
                "true_label":"unknown",
                "true_word":"unknown",
                "pred_label":all_words[val_p],
                "file":None
            })
        if set_name == "train":
            for i,p in enumerate(val_pred):
                if val_correct[i]:
                    # only care about the direction of the unit-normed vector
                    open_max_vector = open_max[i,:]
                    normed_open_max_vector = open_max_vector/np.linalg.norm(open_max_vector)
                    AVs[p].append(normed_open_max_vector)
    if set_name == "train":
        MAVs = {}
        DAVs = defaultdict(list)
        MR_MODELS = {}
        for c in AVs.keys():
            MAVs[c] = np.stack(AVs[c]).mean(axis=0)
            for v in AVs[c]:
                DAVs[c].append(spd.cosine(v,MAVs[c]))
            mr = libmr.MR()
            biggest_true_distances = sorted(DAVs[c])[-20:]
            mr.fit_high(biggest_true_distances,len(biggest_true_distances))
            MR_MODELS[c] = mr
        with open("MAVs.pickle","wb") as f:
            pickle.dump(MAVs,f)
        with open("AVs.pickle","wb") as f:
            pickle.dump(AVs,f)
        with open("DAVs.pickle","wb") as f:
            pickle.dump(DAVs,f)
        with open("MR_MODELS.pickle","wb") as f:
            pickle.dump(MR_MODELS,f)


    tf.logging.info("{} Step {} LR {} Accuracy {} Loss {}".format(set_name,step,learning_rate,val_acc,val_loss))
    df_words = all_words if style == "full" else wanted_words

    pd.DataFrame(val_conf_mat,columns=df_words,index=df_words).to_csv("confusion_matrix_{}.csv".format(set_name))

    pd.DataFrame(pred_df_list).to_csv("predictions_{}.csv".format(set_name))

    # get variance for batch normalization layers back to normal
    if set_name != "train": # (because train runs at the end and we want this to stick for the Comp set)
        for _ in range(50):
            feed_dict = get_batch(data_index["train"],batch_size,style=style)
            feed_dict.update({keep_prob: 0.8,learning_rate_ph:learning_rate,is_training_ph: True})
            _,_ = sess.run([update_ops,loss_mean],feed_dict)

    if set_name == "train":
        return MAVs, MR_MODELS
    else:
        return val_loss.calculate(), val_acc.calculate()


data_index, unknown_index = load_train_data(style=style)
speakers = get_speakers(unknown_index,data_index)
bg_data = wl.load_bg_data(sess)

labels_ph = tf.placeholder(tf.int32,(None))
wav_ph = tf.placeholder(tf.float32,(None,sample_rate))
bg_wavs_ph = tf.placeholder(tf.float32,[None,sample_rate])
keep_prob = tf.placeholder(tf.float32) # will be 0.5 for training, 1 for test
learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")
is_training_ph = tf.placeholder(tf.bool)

processed_wavs = pp.tf_preprocess(wav_ph,bg_wavs_ph,is_training_ph)

features = make_features(processed_wavs,is_training_ph,"log-mel")

output_neurons = len(all_words) if style == "full" else len(wanted_words)
final_layer, open_max_layer = newdrive(features,keep_prob,output_neurons,is_training_ph)

probabilities = tf.nn.softmax(final_layer)

loss_mean = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph, logits=final_layer)

total_loss = tf.losses.get_total_loss()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
train_step = tf.train.MomentumOptimizer(learning_rate_ph,0.9).minimize(total_loss)

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
train_writer = tf.summary.FileWriter("logs/train_unknown_overdrive_full_bn_80_keep_update",sess.graph)
val_writer = tf.summary.FileWriter("logs/val_unknown_overdrive_full_bn_80_keep_update",sess.graph)


tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
last_val_loss = 9999999
for i in range(steps):
    if i > 0 and i % 500 == 0:
        learning_rate = 0.9*learning_rate
    feed_dict = get_batch(data_index["train"],batch_size,style=style)
    feed_dict.update({keep_prob: 0.8,learning_rate_ph:learning_rate,is_training_ph: True})
    # now here's where we run the real, convnet part
    if i % 10 == 0:
        _, sum_val,acc_val,loss_val, _ = sess.run([update_ops,merged_summaries,accuracy_tensor,loss_mean,train_step],feed_dict)
        train_writer.add_summary(sum_val,i)
        tf.logging.info("Step {} LR {} Accuracy {} Cross Entropy {}".format(i,learning_rate,acc_val,loss_val))
    else:
        sess.run(train_step,feed_dict)

    if i % eval_step == 0 or i == (steps - 1):
        val_loss, val_acc = run_validation("val",i)
        if val_loss > last_val_loss:
            learning_rate = 0.75*learning_rate
            print("CHANGING LEARNING RATE TO: {}".format(learning_rate))
            # print("Restoring former model and rerunning validation")
            # saver.restore(sess,"./model.ckpt")
            # val_acc = run_validation("val",i)
        # else:
        #     saver.save(sess,"./model.ckpt")

        last_val_loss = val_loss

    if learning_rate < 0.00001: # at this point, just stop
        saver.save(sess,"./model.ckpt")
        break
test_loss, test_acc = run_validation("test",i)
MAVs, MR_MODELS = run_validation("train",i)

test_index = wl.load_test_data(sess)
# now here's where we run the test classification
import pandas as pd
df = pd.DataFrame([],columns=["fname","label"])
offset = 0
test_batch_size = batch_size
test_info = []
while offset < len(test_index):
    print(offset)
    feed_dict = get_batch(test_index,test_batch_size,offset=offset,mode="comp",style=style)
    feed_dict.update({ keep_prob:1.0,is_training_ph:False})
    test_av, test_prob, test_pred = sess.run([open_max_layer,probabilities,predictions],feed_dict=feed_dict)

    # compute and save distances and MR models probabilities
    for e,p in enumerate(test_pred):
        cos_dist = spd.cosine(test_av[e],MAVs[p])
        unknown_prob = MR_MODELS[p].w_score(cos_dist)
        class_prob = test_prob[e].max()
        fname = test_index[offset + e]["identifier"]
        test_info.append({"MAV_distance":cos_dist,"MAV_prob":unknown_prob,"Class_Prob":class_prob,"Class_Guess":p,"File":fname})


    test_labels = [all_words[test_index] for test_index in test_pred]
    for i in range(len(test_labels)):
        if test_labels[i] not in wanted_words:
            test_labels[i] = "unknown"
    test_files = [t["identifier"] for t in test_index[offset:(offset + test_batch_size)] ]
    test_batch_df = pd.DataFrame([{"fname":ti,"label":tl} for ti,tl in zip(test_files,test_labels)])
    offset += test_batch_size
    df = pd.concat([df,test_batch_df])
df.to_csv("my_guesses_3.csv",index=False)
pd.DataFrame(test_info).to_csv("test_mav_info.csv")
sess.close()

from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "AC7ef9b2470e5800d2cf47640564e18f3f"
# Your Auth Token from twilio.com/console
auth_token  = "e36bb44f5528a0bcbe45509b113d9469"

client = Client(account_sid, auth_token)
message = client.messages.create(
    to="+19082682005",
    from_="+12673607895",
    body="Model finished running with {} val accuracy".format(val_acc)) 

print(message.sid)

