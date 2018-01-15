#!/usr/bin/env bash

python tom_train_1.py --features identity --model simple1d --save simple1d --train --no-val
python tom_train_1.py --features identity --model simple1d --save simple1d_pl1 --train --no-val --pseudo_labels simple1d



