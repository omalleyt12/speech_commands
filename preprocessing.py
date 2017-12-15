import numpy as np
from toolz.functoolz import memoize

def pad(d):
    max_pad = 100
    pad_num = np.random.randint(-max_pad,max_pad)
    if pad_num > 0:
        b = np.pad(d,(pad_num,0),mode="constant")[:-pad_num]
    else:
        b = np.pad(d,(0,-pad_num),mode="constant")[-pad_num:]
    return d

def add_noise(d,bg_data):
    background_frequency = 0.8
    max_background_volume = 0.1
    bg_index = np.random.randint(len(bg_data))
    bg_samp = bg_data[bg_index]
    bg_offset = np.random.randint(0,len(bg_samp) - sample_rate)
    bg_sliced = bg_samp[bg_offset:(bg_offset + sample_rate)]
    if np.random.uniform(0,1) < background_frequency:
        bg_volume = np.random.uniform(0,max_background_volume)
    else:
        bg_volume = 0
    return d + bg_volume*bg_sliced

def reverse(d):
    return d[::-1]

def add(d1,d2):
    return (d1 + d2)/2.0

@memoize
def get_sigmoid_blender(num):
    a = np.linspace(-50,50,num=num)
    return 1 / (1 + np.exp(-a))

def combine(d1,d2):
    sb = get_sigmoid_blender(d1.shape[0])
    return d1*sb + d2*(1 - sb)
