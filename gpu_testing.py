import tensorflow as tf
import tom_shape_ops as shape_ops
from tom_spectral_ops import stft, inverse_stft
import numpy as np
from tensorflow.python.ops import spectral_ops


f = tf.placeholder(tf.float32,[None,16000])
frames = tf.placeholder(tf.float32,[None,1000])

def tf_time_stretch(wav):
    # speedx = tf.truncated_normal([],1,0.2)
    speedx = 1.1
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    s = stft(wav,frame_length,frame_step_in)
    a = inverse_stft(s,frame_length,frame_step_out)
    # return tf_get_word(a,16000)
    return a

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

a = tf.map_fn(tf_time_stretch,f,parallel_iterations=120)

a_batch = tf_batch_time_stretch(f)

b = spectral_ops.rfft(frames,[512])
s = tf.contrib.signal.stft(f,300,100)
si = tf.contrib.signal.inverse_stft(s,300,100)

test_tf_shape = tf.shape(b)

# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

from time import time
start = time()
for _ in range(10):
    # sess.run(test_shape,{frames:np.zeros((1000,1000),dtype=np.float32)})
    # sess.run(s,{f:np.zeros((120,16000),dtype=np.float32)})
    # sess.run(b,{frames:np.zeros((1000,1000),dtype=np.float32)})
    sess.run(a_batch,{f:np.zeros((120,16000),dtype=np.float32)})
print("Time taken")
print(time() - start)
