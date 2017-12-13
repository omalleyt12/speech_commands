# import winsound
import pandas as pd
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
import scipy.io.wavfile as sciwav

def play(a):
    print(a.shape)
    sciwav.write("testing.wav",a.shape[0],a)


batch_size = 100
eval_step = 500
training_step_list = [15000,3000] #[15000,3000]
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

if True:
    test_dir = "test/audio"
    test_index = []
    counter = 0
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_ph = tf.placeholder(tf.string,[])
        wav_loader = io_ops.read_file(wav_filename_ph)
        wav_decoder = contrib_audio.decode_wav(wav_loader,desired_channels=1)
        for wav_path in gfile.Glob(test_dir + "/*.wav"):
            if counter % 1000 == 0:
                print("Test {}".format(counter))
            counter += 1
            file_name = os.path.basename(wav_path)
            tdata = sess.run(wav_decoder,feed_dict={wav_filename_ph:wav_path}).audio.flatten()
            test_index.append({"file":wav_path,"identifier":file_name,"data":tdata})

sess = tf.InteractiveSession()

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
        data_index[set_name].append({"train_label":word,"file":wav_path})

    other_words = list(set(all_words.keys()).difference(set(wanted_words)))
    total_word_list = wanted_words + other_words
    tomIndex = {w:i for i,w in enumerate(total_word_list)} # this will be used for new labeling criteriay

    for k in data_index.keys():
        for i,rec in enumerate(data_index[k]):
            word = data_index[k][i]["train_label"]
            train_label_array = np.zeros(len(total_word_list))
            val_label_array = np.zeros(len(total_word_list))
            train_label_array[tomIndex[word]] = 1
            if word not in wanted_words:
                val_label_array[tomIndex["unknown"]] = 1
                data_index[k][i]["val_label"] = "unknown"
            else:
                val_label_array[tomIndex[word]] = 1
                data_index[k][i]["val_label"] = word
            data_index[k][i]["train_label_array"] = train_label_array
            data_index[k][i]["val_label_array"] = val_label_array

    silence_wav_path = data_index["train"][0]["file"] # arbitrary, to be used for silence
    for set_name in ['val','test','train']:
        # add silence to val, train, test
        set_size = len(data_index[set_name])
        silence_size = int(math.ceil(set_size * silence_percentage) / 100)
        for _ in range(silence_size):
            label_array = np.zeros(len(total_word_list))
            label_array[0] = 1
            data_index[set_name].append({"train_label_array":label_array,"val_label_array":label_array,"train_label":"silence","val_label":"silence","file":silence_wav_path})

    # randomize each set and load data in-mem
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_ph = tf.placeholder(tf.string,[])
        wav_loader = io_ops.read_file(wav_filename_ph)
        wav_decoder = contrib_audio.decode_wav(wav_loader,desired_channels=1,desired_samples=sample_rate)

        for set_name in ['val','test','train']:
            random.shuffle(data_index[set_name])
            for i, rec in enumerate(data_index[set_name]):
                # if i > 100:
                #     break
                if i % 1000 == 0:
                    print("{} {}".format(set_name,i))
                wav_path = data_index[set_name][i]["file"]
                data_index[set_name][i]["data"] = sess.run(wav_decoder,feed_dict={wav_filename_ph:wav_path}).audio.flatten()

    return data_index, total_word_list, tomIndex

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

wav_ph = tf.placeholder(tf.float32,[None,sample_rate])
volume_ph = tf.placeholder(tf.float32,[None])
time_shift_padding_ph = tf.placeholder(tf.int32,[None,2,2])
time_shift_offset_ph = tf.placeholder(tf.int32,[None,2])
bg_ph = tf.placeholder(tf.float32,[None,sample_rate,1])
bg_volume_ph = tf.placeholder(tf.float32,[None])

# this will handle all distortion in parallel, and let me run the graph all together
def distort_wav(ph_tup):
    wav_ph, volume_ph, time_shift_padding_ph, time_shift_offset_ph, bg_ph, bg_volume_ph = ph_tup
    wav = tf.reshape(wav_ph,[sample_rate,1])
    scaled_wav = tf.multiply(wav, volume_ph)
    padded_wav = tf.pad(scaled_wav,time_shift_padding_ph,mode="CONSTANT")
    sliced_wav = tf.slice(padded_wav,time_shift_offset_ph,[sample_rate,-1])
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
        sample_rate,
        dct_coefficient_count = dct_coefficient_count
    )
    mfcc_2d = tf.reshape(mfcc,[mfcc.shape[1],mfcc.shape[2]])
    return mfcc_2d

mfcc = tf.map_fn(
    distort_wav,
    [wav_ph, volume_ph, time_shift_padding_ph, time_shift_offset_ph, bg_ph, bg_volume_ph],
    dtype=tf.float32,
    parallel_iterations=100,
    back_prop=False
 )

def get_mfcc_and_labels(data_index,batch_size,sess,tom_words,tom_index,offset=0,mode="train",return_labels=True):
    labels = []
    if offset + batch_size > len(data_index):
        batch_size = len(data_index) - offset

    ts_pad_list = []
    ts_off_list = []
    vol_list = []
    bg_vol_list = []
    bg_list = []
    wav_list = []
    train_indices_list = []
    val_indices_list = []
    for j in range(batch_size):
        if mode == "train":
            samp_index = np.random.randint(len(data_index))
        else:
            samp_index = offset
            offset += 1

        samp_data = data_index[samp_index]
        wav_list.append(samp_data["data"])

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
        ts_pad_list.append(np.array(time_shift_padding))
        ts_off_list.append(np.array(time_shift_offset))

        # add in background for training, or for test/val that are silence
        if mode == "train" or (return_labels and samp_data["train_label"] == "silence"):
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
        bg_list.append(bg_sliced)
        bg_vol_list.append(bg_volume)

        if "train_label" in samp_data and samp_data["train_label"] == "silence":
            volume = 0 # zero out the foreground for the silence labels
        else:
            volume = 1
        vol_list.append(volume)

        if return_labels:
            labels.append(samp_data["train_label_array"])
            train_indices_list.append(tom_index[samp_data["train_label"]])
            val_indices_list.append(tom_index[samp_data["val_label"]])

    feed_dict={
        wav_ph: np.stack(wav_list),
        volume_ph: np.stack(vol_list),
        time_shift_padding_ph: np.stack(ts_pad_list),
        time_shift_offset_ph: np.stack(ts_off_list),
        bg_ph: np.stack(bg_list),
        bg_volume_ph: np.stack(bg_vol_list)
    }
    # data = sess.run(mfcc,feed_dict=feed_dict)
    if return_labels:
        feed_dict.update({
            train_labels_ph: np.stack(labels),
            train_indices_ph: np.stack(train_indices_list),
            val_indices_ph: np.stack(val_indices_list)
        })
    return feed_dict


data_index, tom_words, tom_index = load_train_data()
bg_data = prepare_background_data()

train_labels_ph = tf.placeholder(tf.float32,[None,len(tom_words)],name="labels_ph")
train_indices_ph = tf.placeholder(tf.int32,[None]) # used for train accuracy
val_indices_ph = tf.placeholder(tf.int32,[None]) # used for val accuracy

keep_prob = tf.placeholder(tf.float32,name="keep_prob") # will be 0.5 for training, 1 for test

fingerprint_4d = tf.reshape(mfcc,[-1,mfcc.shape[1],mfcc.shape[2],1])

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


weights_3 = tf.Variable(tf.truncated_normal([now_flat_elements,len(tom_words)],stddev=0.01))
bias_3 = tf.Variable(tf.zeros([len(tom_words)]))
final_layer = tf.matmul(flat_layer,weights_3) + bias_3

# using sigmoid here!! (bc classes are no longer mutually exclusive)
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_ph, logits=final_layer)
cross_entropy_mean = tf.reduce_mean(cross_entropy_loss)

learning_rate_ph = tf.placeholder(tf.float32,[],name="learning_rate_ph")
train_step = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cross_entropy_mean)

unknown_val = final_layer[:,1] # bc unknown is the second label for all unknown words

# otherwise, unknown is basically always predicted
def map_layer_to_train_pred(x):
    return tf.cond(tf.equal(tf.argmax(x),1),lambda: tf.nn.top_k(x,2)[1][1], lambda: tf.argmax(x,output_type=tf.int32) )
train_predicted_indices = tf.map_fn(map_layer_to_train_pred,final_layer,parallel_iterations=100,dtype=tf.int32)


# map words not in wanted_words back to unknown label
def map_layer_to_val_pred(x):
    return tf.cond(tf.greater_equal(tf.argmax(x),len(wanted_words)),lambda: 1, lambda: tf.argmax(x,output_type=tf.int32) )
val_predicted_indices = tf.map_fn(map_layer_to_val_pred,final_layer,parallel_iterations=100,dtype=tf.int32)

train_correct_prediction = tf.equal(train_predicted_indices,train_indices_ph)
val_correct_prediction = tf.equal(val_predicted_indices,val_indices_ph)

# confusion_matrix = tf.confusion_matrix(val_indices,predicted_indices,num_classes=label_elements)

# don't take unknown into account at all for this
train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction,tf.float32))
# take unknown into account, and don't take any unknown words into account
val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction,tf.float32))

global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step,global_step + 1)

saver = tf.train.Saver(tf.global_variables())

tf.summary.scalar("cross_entropy",cross_entropy_mean)
tf.summary.scalar("train_accuracy",train_accuracy)
tf.summary.scalar("val_accuracy",val_accuracy)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
val_writer = tf.summary.FileWriter("logs/val",sess.graph)

tf.logging.set_verbosity(tf.logging.INFO)
sess.run(tf.global_variables_initializer())


current_step = 0
sum_steps = sum(training_step_list)
for steps, learning_rate in zip(training_step_list,learning_rate_list):
    for i in range(steps):
        current_step += 1
        feed_dict = get_mfcc_and_labels(data_index["train"],batch_size,sess,tom_words,tom_index)
        feed_dict.update({
            learning_rate_ph: learning_rate,
            keep_prob: 0.5
        })
        # now here's where we run the real, convnet part
        if current_step % 10 == 0:
            t_ind,train_summary, t_accuracy, cross_ent_val, _, _ = sess.run(
                [train_predicted_indices,merged_summaries,train_accuracy,cross_entropy_mean,train_step,increment_global_step],
                feed_dict=feed_dict
            )
            train_writer.add_summary(train_summary,current_step)
            tf.logging.info("Step {} Rate {} Accuracy {} Cross Entropy {}".format(current_step,learning_rate,t_accuracy,cross_ent_val))
        else:
            _, _ = sess.run([train_step,increment_global_step],feed_dict)

        if current_step % eval_step == 0 or current_step == sum_steps:
            set_name = "val" if not current_step == sum_steps else "test"
            # run validation loop
            val_size = len(data_index[set_name])
            val_offset = 0
            batches = 0
            v_acc_total = 0
            t_acc_total = 0
            loss_total = 0
            while val_offset < val_size:
                feed_dict = get_mfcc_and_labels(data_index[set_name],batch_size,sess,tom_words,tom_index,offset=val_offset,mode="val")
                feed_dict.update({keep_prob:1.0})
                val_summary, t_accuracy, v_accuracy, val_cem = sess.run([merged_summaries,train_accuracy,val_accuracy,cross_entropy_mean],
                                                            feed_dict=feed_dict
                )
                val_offset += batch_size
                batches += 1
                v_acc_total += v_accuracy
                t_acc_total += t_accuracy
                loss_total += val_cem
            v_acc_avg = v_acc_total / max(batches,1) # for when i don't use a val or test set, don't throw error
            t_acc_avg = t_acc_total / max(batches,1)
            loss_avg = loss_total / max(batches,1)
            val_writer.add_summary(val_summary,current_step)
            tf.logging.info("{} Train Accuracy {} Val Accuracy {} Loss {}".format(set_name,t_acc_avg,v_acc_avg,loss_avg))

if True:
    # now here's where we run the test classification
    import pandas as pd
    df = pd.DataFrame([],columns=["fname","label"])

    offset = 0
    test_batch_size = 100
    while offset < len(test_index):
        feed_dict = get_mfcc_and_labels(test_index,test_batch_size,sess,tom_words,tom_index,offset=offset,mode="test",return_labels=False)
        feed_dict.update({keep_prob:1.0})
        test_pred = sess.run(val_predicted_indices,feed_dict=feed_dict)
        test_labels = [tom_words[tind] for tind in test_pred]

        test_files = [t["identifier"] for t in test_index[offset:(offset + test_batch_size)] ]

        test_batch_df = pd.DataFrame([{"fname":ti,"label":tl} for ti,tl in zip(test_files,test_labels)])
        offset += test_batch_size

        print(offset)
        df = pd.concat([df,test_batch_df])

    df.to_csv("my_guesses_3.csv",index=False)
