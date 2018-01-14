"""
Create a prediction out of an ensemble of model predictions
"""
import os
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

wanted_words = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]
clips = defaultdict(list)

for f in os.listdir("ensemble_models"):
    if ".pickle" in f:
        with open("ensemble_models/" + f,"rb") as f:
            test_info = pickle.load(f)
            for d in test_info:
                clips[d["fname"]].append(d["all_prob"])

pred_list = []
for fname, prob_list in clips.items():
    guess = np.power(np.stack(prob_list),2).mean(axis=0).argmax()
    pred_list.append({
        "fname":fname,
        "label":wanted_words[guess]
    })

pd.DataFrame(pred_list).to_csv("ensemble_prediction.csv",index=False)



