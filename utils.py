import os
import re
import hashlib
import random
from tensorflow.python.util import compat

def which_set(wav_path):
    percent_test = 10 # test set
    percent_val = 10 # val set
    """Use hashing of the first part of file name to keep a speaker confined to one set"""
    wav_name = re.sub(r'_nohash_.*$','',os.path.basename(wav_path)) # so that all samps from same user are grouped together
    hash_name_hashed = hashlib.sha1(compat.as_bytes(wav_name)).hexdigest()
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
    percentage_hash = ((int(hash_name_hashed, 16) %
                    (MAX_NUM_WAVS_PER_CLASS + 1)) *
                    (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < percent_val:
        result = 'val'
    elif percentage_hash < (percent_val + percent_test):
        result = 'test'
    else:
        result = 'train'
    return result
