#!/usr/bin/env bash

python tom_train_1.py --features log-mel --model small_overdrive --save pi_small_overdrive_bigger --train --no-val --pseudo_labels overdrive
python tom_train_1.py --features log-mel --model newdrive --save  pi_newdrive_bigger --train --no-val --pseudo_labels overdrive



