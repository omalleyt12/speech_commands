#!/usr/bin/env bash
python tom_train_1.py --features log-mel --model ap_overdrive --save ap_overdrive --train --no-val
python tom_train_1.py --features log-mel --model ap_overdrive --save ap_overdrive_pl1 --pseudo_labels ap_overdrive --train --no-val
python tom_train_1.py --features log-mel --model ap_overdrive --save ap_overdrive_pl2 --pseudo_labels ap_overdrive_pl1 --train --no-val

