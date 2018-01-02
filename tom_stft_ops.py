
import tensorflow as tf
import numpy as np
from tensorflow.contrib.signal.python.ops import window_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.python.ops import spectral_ops
from tensorflow.contrib.signal.python.ops import reconstruction_ops

def stft(signals,frame_length,frame_step):
    hann_window = window_ops.hann_window(frame_length)
    framed_signals = shape_ops.frame(signals, frame_length, frame_step,pad_end=False)
    framed_signals *= hann_window
    return spectral_ops.rfft(framed_signals)

def inverse_stft(stfts,frame_length,frame_step):
    real_frames = spectral_ops.irfft(stfts)
    hann_window = window_ops.hann_window(frame_length)
    return reconstruction_ops.overlap_and_add(real_frames, frame_step)

def fast_time_stretch(signals):
    def overlap(tup):
        framed_signals, frame_step_out = tup
        new_wav = reconstruction_ops.overlap_and_add(framed_signals,frame_step_out)
        return tf_get_word(new_wav)

    speedx = tf.truncated_normal([tf.shape(signals)[0]],1,0.2)
    frame_length = 300
    frame_step_in = int(300*0.25)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    hann_window = window_ops.hann_window(frame_length)
    framed_signals = shape_ops.frame(signals, frame_length, frame_step_in,pad_end=False)
    framed_signals *= hann_window
    return tf.map_fn(overlap,[framed_signals,frame_step_out],parallel_iterations=120,back_prop=False,dtype=tf.float32,infer_shape=False)

def fast_pitch_shift(signals):
    def resample(tup):
        framed_signals, frame_step_out = tup
        new_wav = reconstruction_ops.overlap_and_add(framed_signals,frame_step_out)
        return tf_resample(new_wav)

    speedx = tf.truncated_normal([tf.shape(signals)[0]],1,0.2)
    frame_length = 300
    frame_step_in = int(300*0.25)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    hann_window = window_ops.hann_window(frame_length)
    framed_signals = shape_ops.frame(signals, frame_length, frame_step_in,pad_end=False)
    framed_signals *= hann_window
    return tf.map_fn(resample,[framed_signals,frame_step_out],parallel_iterations=120,back_prop=False,dtype=tf.float32,infer_shape=False)

def tf_time_stretch(wav):

    speedx = tf.truncated_normal([],1,0.2)
    frame_length = 300
    frame_step_in = int(300*0.25)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    s = stft(wav,frame_length,frame_step_in)
    a = inverse_stft(s,frame_length,frame_step_out)
    return tf_get_word(a,16000)

def tf_batch_time_stretch(wavs):
    speedx = tf.truncated_normal([],1,0.2)
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    s = stft(wavs,frame_length,frame_step_in)
    a = inverse_stft(s,frame_length,frame_step_out)
    return tf.map_fn(tf_get_word,wavs,parallel_iterations=120,back_prop=False)

def tf_get_word(wav,size=16000):
    frames = shape_ops.frame(wav,size,300,pad_end=True)
    frame_stack = tf.stack(frames)
    frame_vols = tf.reduce_mean(tf.pow(frame_stack,2),axis=1)
    max_frame_vol = tf.argmax(frame_vols)
    return frame_stack[max_frame_vol,:]

f = tf.placeholder(tf.float32,[None,16000])
a = tf.map_fn(tf_time_stretch,f,parallel_iterations=120)

s = stft(f,30*16,10*16)
si = inverse_stft(s,30*16,5*16)

ts = time_stretch(f)

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

from time import time
start = time()
for _ in range(10):
    # sess.run(test_shape,{frames:np.zeros((1000,1000),dtype=np.float32)})
    # sess.run(s,{f:np.zeros((120,16000),dtype=np.float32)})
    # sess.run(b,{frames:np.zeros((1000,1000),dtype=np.float32)})
    sess.run(a,{f:np.zeros((100,16000),dtype=np.float32)})
print("Time taken")
print(time() - start)








