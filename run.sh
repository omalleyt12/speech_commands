#!/usr/bin/env bash
python tom_train_1.py --features long-log-mel --model overdrive --save overdrive_long --train --no-val
python tom_train_1.py --features long-log-mel --model overdrive --save overdrive_long_pl1 --pseudo_labels overdrive_long --train --no-val
python tom_train_1.py --features long-log-mel --model overdrive --save overdrive_long_pl2 --pseudo_labels overdrive_long_pl1 --train --no-val

