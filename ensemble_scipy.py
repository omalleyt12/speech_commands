import pandas as pd
import pickle
from collections import defaultdict
from itertools import product
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import argparse

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
    train_input.append(np.stack(data_row))
    train_label.append(label_row)

train_input = np.stack(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.int32)

def loss_func(weights):
    weights = weights.reshape((10,-1))
    for i in range(weights.shape[1]):
        np.nonzero(train_label == i+2)



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

