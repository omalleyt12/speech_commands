import re
import math
import os
import hashlib
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.util import compat

sess = tf.InteractiveSession()

batch_size = 100
eval_step = 500
training_step_list = [15000,3000]
learning_rate_list = [0.001,0.0001]
data_dir = "train/audio"
summary_dir = "logs" # where to save summary logs for Tensorboard
wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
sample_rate = 16000 # per sec
clip_duration_ms = 1000 #
max_background_volume = 0.1 # how loud background noise should be, [0,1]
background_frequency = 0.8 # how many of training samps have noise added
silence_percentage = 10.0 # what percent of training data should be silence
unknown_percentage = 10.0 # what percent of training data should be unknown words
max_time_shift_ms = 100.0 # max range to randomly shift the training audio by time
window_size_ms = 30.0 # millisec length of frequency analysis window
window_stride_ms = 10.0
window_size_samples = int(sample_rate * window_size_ms / 1000)
window_stride_samples = int(sample_rate * window_stride_ms / 1000)
dct_coefficient_count = 40 # bins to use for MFCC fingerprint
percent_test = 10 # test set
percent_val = 10 # val set

def which_set(wav_path):
    # split into train, test, val sets deterministically
    wav_name = re.sub(r'_nohash_.*$','',os.path.basename(wav_path)) # so that all samps from same user are grouped together
    hash_name_hashed = hashlib.sha1(compat.as_bytes(wav_name)).hexdigest()
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
    percentage_hash = ((int(hash_name_hashed, 16) %
                    (MAX_NUM_WAVS_PER_CLASS + 1)) *
                    (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < percent_val:
        result = 'val'
    elif percentage_hash < (percent_val + percent_test):
        result = 'test'
    else:
        result = 'train'
    return result

def load_train_data():
    random.seed(111)
    wanted_words_index = {}
    for i, wanted_word in enumerate(wanted_words):
        wanted_words_index[wanted_word] = i

    data_index = {"val":[],"test":[],"train":[]}
    unknown_index = {"val":[],"test":[],"train":[]}

    all_words = {}

    for wav_path in gfile.Glob(data_dir + "/*/*.wav"):
        word = os.path.split(os.path.dirname(wav_path))[-1].lower()
        if word == "_background_noise_":
            continue # don't include yet

        all_words[word] = True

        set_name = which_set(wav_path)

        if word in wanted_words:
            data_index[set_name].append({"label":word,"file":wav_path})
        else:
            unknown_index[set_name].append({"label":word,"file":wav_path})

    silence_wav_path = data_index["train"][0]["file"] # arbitrary, to be used for silence
    for set_name in ['val','test','train']:
        # add silence to val, train, test
        set_size = len(data_index[set_name])
        silence_size = int(math.ceil(set_size * silence_percentage) / 100)
        for _ in range(silence_size):
            data_index[set_name].append({"label":"silence","file":silence_wav_path})

        # add unknown words to val, train, test
        random.shuffle(unknown_index[set_name])
        unknown_size = int(math.ceil(set_size * unknown_percentage) / 100)
        data_index[set_name].extend(unknown_index[set_name][:unknown_size])

    # randomize each set
    for set_name in ['val','test','train']:
        random.shuffle(data_index[set_name])

    # get word2index mapping
    word2index = {}
    for word in all_words:
        if word in wanted_words_index:
            word2index[word] = wanted_words_index[word]
        else:
            word2index[word] = 1
    word2index["silence"] = 0

    return data_index, word2index

def prepare_background_data():
    bg_data = []
    bg_path = "train/audio/_background_noise_/*.wav"
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_ph = tf.placeholder(tf.string,[])
        wav_loader = io_ops.read_file(wav_filename_ph)
        wav_decoder = contrib_audio.decode_wav(wav_loader,desired_channels=1)
        for wav_path in gfile.Glob(bg_path):
            wav_data = sess.run(wav_decoder,feed_dict={wav_filename_ph:wav_path}).audio.flatten()
            bg_data.append(wav_data)
    return bg_data


# Preprocessing Graph parts
wav_filename_ph = tf.placeholder(tf.string,[])
wav_loader = io_ops.read_file(wav_filename_ph)
wav_decoder = contrib_audio.decode_wav(wav_loader,desired_channels=1,desired_samples=sample_rate) # length of 1 sec

volume_ph = tf.placeholder(tf.float32,[])
scaled_wav = tf.multiply(wav_decoder.audio, volume_ph)

time_shift_padding_ph = tf.placeholder(tf.int32,[2,2])
time_shift_offset_ph = tf.placeholder(tf.int32,[2])

padded_wav = tf.pad(scaled_wav,time_shift_padding_ph,mode="CONSTANT")
sliced_wav = tf.slice(padded_wav,time_shift_offset_ph,[sample_rate,-1])

bg_ph = tf.placeholder(tf.float32,[sample_rate,1])
bg_volume_ph = tf.placeholder(tf.float32,[])

scaled_bg = tf.multiply(bg_ph,bg_volume_ph)
wav_with_bg = tf.add(sliced_wav,scaled_bg)
clamped_wav = tf.clip_by_value(wav_with_bg,-1.0,1.0)

spectrogram = contrib_audio.audio_spectrogram(
    clamped_wav,
    window_size = window_size_samples,
    stride = window_stride_samples,
    magnitude_squared = True
)
mfcc = contrib_audio.mfcc(
    spectrogram,
    wav_decoder.sample_rate,
    dct_coefficient_count = dct_coefficient_count
)

def get_mfcc_and_labels(data_index,batch_size,sess,word2index,offset=0,mode="train",return_labels=True):
    data = []
    labels = []
    if offset + batch_size > len(data_index):
        batch_size = len(data_index) - offset

    for j in range(batch_size):
        if mode == "train":
            samp_index = np.random.randint(len(data_index))
        else:
            samp_index = offset
            offset += 1

        samp_data = data_index[samp_index]

        if mode == "train":
            time_shift_amount = np.random.randint(-max_time_shift_ms,max_time_shift_ms)
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount,0],[0,0]]
                time_shift_offset = [0,0]
            else:
                time_shift_padding = [[0,-time_shift_amount],[0,0]]
                time_shift_offset = [-time_shift_amount,0]
        else:
            time_shift_padding = [[0,0],[0,0]]
            time_shift_offset = [0,0]

        # add in background for training, or for test/val that are silence
        if mode == "train" or (return_labels and samp_data["label"] == "silence"):
            bg_index = np.random.randint(len(bg_data))
            bg_samp = bg_data[bg_index]
            bg_offset = np.random.randint(0,len(bg_samp) - sample_rate)
            bg_sliced = bg_samp[bg_offset:(bg_offset + sample_rate)]
            bg_sliced = bg_sliced.reshape(sample_rate,-1)
            if np.random.uniform(0,1) < background_frequency:
                bg_volume = np.random.uniform(0,max_background_volume)
            else:
                bg_volume = 0
        else:
            bg_sliced = np.zeros((sample_rate,1))
            bg_volume = 0

        if "label" in samp_data and samp_data["label"] == "silence":
            volume = 0 # zero out the foreground for the silence labels
        else:
            volume = 1

        feed_dict={
            wav_filename_ph: samp_data["file"],
            volume_ph: volume,
            time_shift_padding_ph: time_shift_padding,
            time_shift_offset_ph: time_shift_offset,
            bg_ph: bg_sliced,
            bg_volume_ph: bg_volume
        }
        data.append(sess.run(mfcc,feed_dict=feed_dict))
        if return_labels:
            samp_label = np.zeros(len(wanted_words))
            samp_label[word2index[samp_data["label"]]] = 1
            labels.append(samp_label)

    data = np.concatenate(data,axis=0)
    if not return_labels:
        return data
    else:
        labels = np.stack(labels,axis=0)
        return data, labels


label_elements = len(wanted_words)

mfcc_height = mfcc.shape[1]
mfcc_width = mfcc.shape[2]
fingerprint_ph = tf.placeholder(tf.float32,[None,mfcc_height,mfcc_width],name="fingerprint_ph")
labels_ph = tf.placeholder(tf.float32,[None,label_elements],name="labels_ph")

keep_prob = tf.placeholder(tf.float32,name="keep_prob") # will be 0.5 for training, 1 for test

fingerprint_4d = tf.reshape(fingerprint_ph,[-1,mfcc_height,mfcc_width,1])

conv_1_channels = 64
conv_2_channels = 64

weights_1 = tf.Variable(tf.truncated_normal([20,8,1,conv_1_channels],stddev=0.01)) # [height,width,depth,channels]
bias_1 = tf.Variable(tf.zeros([64]))
conv_1 = tf.nn.conv2d(fingerprint_4d,weights_1,[1,1,1,1],"SAME") + bias_1
relu_1 = tf.nn.relu(conv_1)
dropout_1 = tf.nn.dropout(relu_1,keep_prob)
maxpool_1 = tf.nn.max_pool(dropout_1,[1,2,2,1],[1,2,2,1],"SAME")

weights_2 = tf.Variable(tf.truncated_normal([10,4,conv_1_channels,conv_2_channels],stddev=0.01)) # [height,width,first_conv_channels,second_conv_channels]
bias_2 = tf.Variable(tf.zeros([64]))
conv_2 = tf.nn.conv2d(maxpool_1,weights_2,[1,1,1,1],"SAME") + bias_2
relu_2 = tf.nn.relu(conv_2)
dropout_2 = tf.nn.dropout(relu_2,keep_prob)

_ , now_height, now_width, _ = dropout_2.get_shape()
now_height = int(now_height)
now_width = int(now_width)
now_flat_elements = now_height * now_width * conv_2_channels

flat_layer = tf.reshape(dropout_2,[-1,now_flat_elements])


weights_3 = tf.Variable(tf.truncated_normal([now_flat_elements,label_elements],stddev=0.01))
bias_3 = tf.Variable(tf.zeros([label_elements]))
final_layer = tf.matmul(flat_layer,weights_3) + bias_3

cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_ph, logits=final_layer)
cross_entropy_mean = tf.reduce_mean(cross_entropy_loss)

learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")
train_step = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cross_entropy_mean)

predicted_indices = tf.argmax(final_layer,1)
actual_indices = tf.argmax(labels_ph,1)

correct_prediction = tf.equal(predicted_indices,actual_indices)
confusion_matrix = tf.confusion_matrix(actual_indices,predicted_indices,num_classes=label_elements)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step,global_step + 1)

saver = tf.train.Saver(tf.global_variables())

tf.summary.scalar("cross_entropy",cross_entropy_mean)
tf.summary.scalar("accuracy",accuracy)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
val_writer = tf.summary.FileWriter("logs/val",sess.graph)

tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())

data_index, word2index = load_train_data()
bg_data = prepare_background_data()

current_step = 0
sum_steps = sum(training_step_list)
for steps, learning_rate in zip(training_step_list,learning_rate_list):
    for i in range(steps): 
        current_step += 1
        data, labels = get_mfcc_and_labels(data_index["train"],batch_size,sess,word2index)
        # now here's where we run the real, convnet part
        train_summary, train_accuracy, cross_ent_val, _, _ = sess.run(
            [merged_summaries,accuracy,cross_entropy_mean,train_step,increment_global_step],
            feed_dict={fingerprint_ph: data,labels_ph: labels,learning_rate_ph: learning_rate,keep_prob: 0.5}
        )
        train_writer.add_summary(train_summary,current_step)
        tf.logging.info("Step {} Rate {} Accuracy {} Cross Entropy {}".format(current_step,learning_rate,train_accuracy,cross_ent_val))

        if current_step % eval_step == 0 or current_step == sum_steps:
            set_name = "val" if not current_step == sum_steps else "test"
            # run validation loop
            val_size = len(data_index[set_name])
            val_data, val_labels = get_mfcc_and_labels(data_index[set_name],val_size,sess,word2index,mode="val")
            val_summary, val_accuracy, val_cem = sess.run([merged_summaries,accuracy,cross_entropy_mean],
                feed_dict={fingerprint_ph: val_data,labels_ph: val_labels,keep_prob: 1.0}
            )
            val_writer.add_summary(val_summary,current_step)
            tf.logging.info("{} Accuracy {} Loss {}".format(set_name,val_accuracy,val_cem))

# now here's where we run the test classification
import pandas as pd
df = pd.DataFrame([],columns=["fname","label"])

test_dir = "test/audio"
test_index = []
for wav_path in gfile.Glob(test_dir + "/*.wav"):
    file_name = os.path.basename(wav_path) 
    test_index.append({"file":wav_path,"identifier":file_name})
offset = 0
test_batch_size = 1000
while offset < len(test_index):
    test_data = get_mfcc_and_labels(test_index,test_batch_size,sess,word2index,offset=offset,mode="test",return_labels=False)

    test_pred = sess.run(predicted_indices,feed_dict= {fingerprint_ph: test_data,keep_prob:0.5})
    test_labels = [wanted_words[tind] for tind in test_pred]

    test_files = [t["identifier"] for t in test_index[offset:(offset + test_batch_size)] ]

    test_batch_df = pd.DataFrame([{"fname":ti,"label":tl} for ti,tl in zip(test_files,test_labels)])
    offset += test_batch_size
 
    print(offset)
    df = pd.concat([df,test_batch_df])

df.to_csv("my_guesses.csv",index=False)

