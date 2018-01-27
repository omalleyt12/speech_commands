# Instructions for creating submissions file:

The commands to generate the submission file are contained in run.sh in the zip file provided. This guide assumes that the training files have been placed in a directory called “train” under this directory and that the test files are in a directory called “test”.
The commands (contained in run.sh) are also listed below:

````
#!/usr/bin/env bash
python train.py --features log-mel --model overdrive --save overdrive --train --no-val
python train.py --features log-mel --model overdrive --save overdrive_pl1 --pseudo_labels overdrive --train --no-val
python train.py --features log-mel --model overdrive --save overdrive_pl2 --pseudo_labels overdrive_pl1 --train --no-val
python train.py --features equal-log-mel --model overdrive --save overdrive_frame_eq --train --no-val --pseudo_labels overdrive
python train.py --features log-mel-40 --model okconv --save okconv --train --no-val
python train.py --features mfcc-13 --model mfccnet --save mfccnet --train --no-val
python train.py --features mfcc-13 --model mfccnet --save mfccnet_pl1 --train --no-val --pseudo_labels mfccnet
python train.py --features short-log-mel --model overdrive --save overdrive_short --train --no-val
python train.py --features short-log-mel --model overdrive --save overdrive_short_pl1 --pseudo_labels overdrive_short --train --no-val
python train.py --features short-log-mel --model overdrive --save overdrive_short_pl2 --pseudo_labels overdrive_short_pl1 --train --no-val

cp models/overdrive_comp.pickle ensemble_models/
cp models/overdrive_pl1_comp.pickle ensemble_models/
cp models/overdrive_pl2_comp.pickle ensemble_models/
cp models/overdrive_frame_eq_comp.pickle ensemble_models/
cp models/okconv_comp.pickle ensemble_models/
cp models/overdrive_short_comp.pickle ensemble_models/
cp models/overdrive_short_pl1_comp.pickle ensemble_models/
cp models/overdrive_short_pl2_comp.pickle ensemble_models/

python ensemble_sqrt.py
````

