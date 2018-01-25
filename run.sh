#!/usr/bin/env bash
python tom_train_1.py --features log-mel --model overdrive --save overdrive --train --no-val
python tom_train_1.py --features log-mel --model overdrive --save overdrive_pl1 --pseudo_labels overdrive --train --no-val
python tom_train_1.py --features log-mel --model overdrive --save overdrive_pl2 --pseudo_labels overdrive_pl1 --train --no-val
python tom_train_1.py --features equal-log-mel --model overdrive --save overdrive_frame_eq --train --no-val --pseudo_labels overdrive
python tom_train_1.py --features log-mel-40 --model okconv --save okconv --train --no-val
python tom_train_1.py --features mfcc-13 --model mfccnet --save mfccnet --train --no-val
python tom_train_1.py --features mfcc-13 --model mfccnet --save mfccnet_pl1 --train --no-val --pseudo_labels mfccnet
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short --train --no-val
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short_pl1 --pseudo_labels overdrive_short --train --no-val
python tom_train_1.py --features short-log-mel --model overdrive --save overdrive_short_pl2 --pseudo_labels overdrive_short_pl1 --train --no-val

cp models/overdrive_comp.pickle ensemble_models/
cp models/overdrive_pl1_comp.pickle ensemble_models/
cp models/overdrive_pl2_comp.pickle ensemble_models/
cp models/overdrive_frame_eq_comp.pickle ensemble_models/
cp models/okconv_comp.pickle ensemble_models/
cp models/overdrive_short_comp.pickle ensemble_models/
cp models/overdrive_short_pl1_comp.pickle ensemble_models/
cp models/overdrive_short_pl2_comp.pickle ensemble_models/

python ensemble_sqrt.py

