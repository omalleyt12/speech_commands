#!/usr/bin/env bash
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short --train --no-val
python tom_train_1.py --features log-mel --model overdrive --save overdrive_10_pl2 --pseudo_labels overdrive_10_pl1 --train --no-val --seed 10
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short_pl1 --pseudo_labels overdrive_short --train --no-val
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short_pl2 --pseudo_labels overdrive_short_pl1 --train --no-val

