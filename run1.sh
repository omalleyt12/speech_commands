#!/usr/bin/env bash

python tom_train_1.py --features log-mel --model ap_overdrive --save ap_overdrive --train --no-val
python tom_train_1.py --features log-mel --model small_overdrive --save small_overdrive --train --no-val
python tom_train_1.py --features log-mel --model smaller_overdrive --save smaller_overdrive --train --no-val
python tom_train_1.py --features log-mel --model squeeze_overdrive --save squeeze_overdrive --train --no-val



