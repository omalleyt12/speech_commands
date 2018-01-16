#!/usr/bin/env bash

python tom_train_probs.py --features mfcc-13 --model mfccnet --restore mfccnet_pl1
python tom_train_probs.py --features equal-log-mel --model overdrive --restore overdrive_frame_eq
python tom_train_probs.py --features log-mel --model overdrive --restore overdrive_pl2
python tom_train_probs.py --features short-log-mel --model overdrive --restore overdrive_short_pl1
python tom_train_probs.py --features log-mel --model overdrive --restore overdrive
python tom_train_probs.py --features log-mel --model overdrive --restore overdrive_pl1
python tom_train_probs.py --features short-log-mel --model overdrive --restore overdrive_short




