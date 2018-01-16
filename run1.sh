#!/usr/bin/env bash

python tom_train_1.py --features log-mel --model small_overdrive --save pi_new_small_overdrive --train --no-val --pseudo_labels overdrive
python tom_train_1.py --features log-mel --model newdrive --save pi_new_newdrive --train --no-val --pseudo_labels overdrive



