#!/usr/bin/env bash

python tom_train_1.py --features log-mel --model crnn --save crnn --train --no-val
python tom_train_1.py --features log-mel --model crnn --save crnn_pl1 --train --no-val --pseudo_labels crnn
python tom_train_1.py --features log-mel --model crnn --save crnn_pl2 --train --no-val --pseudo_labels crnn_pl1



