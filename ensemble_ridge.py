import pandas as pd
import pickle
from collections import defaultdict
from itertools import product
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--l1',
    type=float,
    default=0.0
)
parser.add_argument(
    '--l2',
    type=float,
    default=0.0
)
FLAGS, unparsed = parser.parse_known_args()

train_guesses = []
models = sorted(os.listdir("train_probs"))
for f in models:
    with open("train_probs/" + f,"rb") as pf:
        train_guesses.append(pickle.load(pf))

train_input = []
train_label = []
for i in range(len(train_guesses[0])):
    data_row = []
    label_row = train_guesses[0][i]["label"]
    for m in train_guesses:
        data_row.append(m[i]["all_prob"])
        if m[i]["label"] != label_row:
            print("Something went wrong")
    train_input.append(np.concatenate(data_row))
    train_label.append(label_row)

train_input = np.stack(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.int32)

learning_rate = tf.placeholder(tf.float32)
input_ph = tf.placeholder(tf.float32,[None,12*len(train_guesses)])
label_ph = tf.placeholder(tf.int32,[None])
l1_ph = tf.placeholder(tf.float32)
l2_ph = tf.placeholder(tf.float32)

# if FLAGS.l1 == 0.0 and FLAGS.l2 == 0:
#     fc = slim.fully_connected(input_ph,12,activation_fn=None,biases_initializer=None)
# else:
fc = slim.fully_connected(input_ph,12,activation_fn=None,biases_initializer=None,weights_regularizer=slim.l1_l2_regularizer(l1_ph,l2_ph))

soft_max = tf.nn.softmax(fc)

cost = tf.losses.sparse_softmax_cross_entropy(label_ph,fc)
total_cost = tf.losses.get_total_loss()

train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)

if False:
    train_size = int(0.8*train_input.shape[0])
    val_input = train_input[train_size:,:]
    val_label = train_label[train_size:]

    train_input = train_input[:train_size,:]
    train_label = train_label[:train_size]

    l1s = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,0]
    l2s = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,0]

    min_val_cost = float('inf')
    opt_l1 = None
    opt_l2 = None
    for l1, l2 in product(l1s,l2s):
        # print("L1 {} L2 {}".format(l1,l2))
        lr = 1
        batch_size = 100
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        cum_c = 0
        for i in range(10000):
            if i % 1000 == 0:
                lr = lr*0.5

            samps = np.random.choice(train_input.shape[0],batch_size,replace=False)
            batch_input = train_input[samps,:]
            batch_labels = train_label[samps]
            w, _, c = sess.run([fc,train_step,cost],{input_ph:batch_input,label_ph:batch_labels,learning_rate:lr,l1_ph:l1,l2_ph:l2})
            cum_c = 0.9*cum_c + 0.1*c
            # if i % 5000 == 0:
            #     print("Step {} Cost {}".format(i,cum_c))

        val_cost = sess.run(cost,{input_ph:val_input,label_ph:val_label})
        print("Validationish Cross Entropy {} for L1 {} and L2 {}".format(val_cost,l1,l2))
        if val_cost < min_val_cost:
            print("Best so far")
            opt_l1 = l1
            opt_l2 = l2
            min_val_cost = val_cost
    print("Optimal L1 {}".format(opt_l1))
    print("Optimal L2 {}".format(opt_l2))

lr = 1
batch_size = 100
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
cum_c = 0
for i in range(10000):
    if i % 1000 == 0:
        lr = lr*0.5

    samps = np.random.choice(train_input.shape[0],batch_size,replace=False)
    batch_input = train_input[samps,:]
    batch_labels = train_label[samps]
    w, _, c = sess.run([fc,train_step,cost],{input_ph:batch_input,label_ph:batch_labels,learning_rate:lr,l1_ph:1e-5,l2_ph:1e-5})
    cum_c = 0.9*cum_c + 0.1*c
    if i % 5000 == 0:
        print("Step {} Cost {}".format(i,cum_c))

if True:
    wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
    clips = defaultdict(list)

    models = sorted(os.listdir("ensemble_models"))
    for f in models:
        if ".pickle" in f:
            with open("ensemble_models/" + f,"rb") as f:
                test_info = pickle.load(f)
                for d in test_info:
                    clips[d["fname"]].append(d["all_prob"])

    pred_list = []
    k = 0
    for fname, prob_list in clips.items(): 
        if k % 10000 == 0:
            print(k)
        k += 1
        test_input = np.concatenate(prob_list).reshape((1,-1))
        ens_prob = sess.run(soft_max,{input_ph:test_input})
        guess = ens_prob[0].argmax()
        pred_list.append({
            "fname":fname,
            "label":wanted_words[guess]
        })

    pd.DataFrame(pred_list).to_csv("ensemble_ridge_prediction.csv",index=False)

    print(pd.DataFrame(pred_list)["label"].value_counts())

