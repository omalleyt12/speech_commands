from numba import jit
import tensorflow as tf
import numpy as np
import scipy as sp
from toolz.functoolz import memoize
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.contrib.signal.python.ops import reconstruction_ops
from tensorflow.contrib.signal.python.ops import window_ops
import acoustics

sample_rate = 16000


def tf_preprocess(wavs,bg_wavs,is_training,slow_down):
    def training_process(wavs,bg_wavs):
        # wavs = fast_pitch_shift(wavs)
        wavs = fast_time_stretch(wavs)
        return tf.map_fn(train_preprocess,[wavs,bg_wavs],parallel_iterations=120,dtype=tf.float32,back_prop=False)

    def testing_process(wavs):
        wavs = tf.cond(slow_down,lambda: fast_time_stretch(wavs,constant=True),lambda: wavs)
        return tf.map_fn(test_preprocess,wavs,parallel_iterations=120,back_prop=False)

    return tf.cond(is_training,lambda: training_process(wavs,bg_wavs), lambda: testing_process(wavs))

def train_preprocess(tensors):
    wav = tensors[0]
    bg_wav = tensors[1]
    wav = tf_pad(wav)
    # wav = tf_volume_equalize(wav) # equalize the volume BEFORE adding noise, trying again
    wav = tf_add_noise(wav,bg_wav)
    return tf_volume_equalize(wav) # equalize the volume again AFTER adding noise

def test_preprocess(wav):
    return tf_volume_equalize(wav)

def fast_time_stretch(signals,constant=False):
    def overlap(tup):
        # framed_signals, frame_step_out, resample_x = tup
        framed_signals, frame_step_out = tup
        new_wav = reconstruction_ops.overlap_and_add(framed_signals,frame_step_out)
        # new_wav = tf_get_word(new_wav,size=tf.cast(16000*resample_x,tf.int32))
        # return tf_resample(new_wav)
        return tf_get_word(new_wav)

    if not constant:
        speedx = tf.truncated_normal([tf.shape(signals)[0]],1.0,0.2)
        # pitch = tf.truncated_normal([tf.shape(signals[0])],0.0,2)
    else:
        speedx = tf.truncated_normal([tf.shape(signals)[0]],1.15,0.0001) # literally can't figure out a way to do tf.repeat
        # pitch = tf.constant(np.repeat(0,signals.shape[0]).astype(np.float32))
    frame_length = 300
    frame_step_in = int(300*0.25)
    # resample_x = 2**(pitch/12)
    # frame_step_out = tf.cast(speedx*resample_x*frame_step_in,tf.int32)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    hann_window = window_ops.hann_window(frame_length)
    framed_signals = shape_ops.frame(signals, frame_length, frame_step_in,pad_end=False)
    framed_signals *= hann_window
    # return tf.map_fn(overlap,[framed_signals,frame_step_out,resample_x],parallel_iterations=120,back_prop=False,dtype=tf.float32,infer_shape=False)
    return tf.map_fn(overlap,[framed_signals,frame_step_out],parallel_iterations=120,back_prop=False,dtype=tf.float32)

def fast_pitch_shift(signals):
    def resample(tup):
        framed_signals, frame_step_out = tup
        new_wav = reconstruction_ops.overlap_and_add(framed_signals,frame_step_out)
        return tf_resample(new_wav)

    pitch = tf.random_uniform([tf.shape(signals)[0]],-3,3)
    frame_length = 300
    frame_step_in = int(300*0.25)
    resample_x = 2**(pitch/12)
    frame_step_out = tf.cast(frame_step_in*resample_x,tf.int32)
    hann_window = window_ops.hann_window(frame_length)
    framed_signals = shape_ops.frame(signals, frame_length, frame_step_in,pad_end=False)
    framed_signals *= hann_window
    return tf.map_fn(resample,[framed_signals,frame_step_out],parallel_iterations=120,back_prop=False,dtype=tf.float32,infer_shape=False)

def tf_pitch_shift(wav):
    pitch = tf.truncated_normal([],0,3)
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    resample_x = 2**(pitch/12)
    frame_step_out = tf.cast(frame_step_in*resample_x,tf.int32)
    s = tf.contrib.signal.stft(wav,frame_length,frame_step_in)
    a = tf.contrib.signal.inverse_stft(s,frame_length,frame_step_out)
    return tf_resample(a)

def tf_batch_pitch_shift(wav):
    pitch = tf.truncated_normal([],0,5)
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    resample_x = 2**(pitch/12)
    frame_step_out = tf.cast(frame_step_in*resample_x,tf.int32)
    s = tf.contrib.signal.stft(wav,frame_length,frame_step_in)
    a = tf.contrib.signal.inverse_stft(s,frame_length,frame_step_out)
    return tf.map_fn(tf_resample,a,parallel_iterations=120,back_prop=False)

def tf_time_stretch(wav):
    speedx = tf.truncated_normal([],1,0.2)
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    s = tf.contrib.signal.stft(wav,frame_length,frame_step_in)
    a = tf.contrib.signal.inverse_stft(s,frame_length,frame_step_out)
    return tf_get_word(a,16000)

def tf_batch_time_stretch(wavs):
    speedx = tf.truncated_normal([],1,0.2)
    frame_length = 300
    frame_step_in = int(0.25 * 300)
    frame_step_out = tf.cast(speedx*frame_step_in,tf.int32)
    s = tf.contrib.signal.stft(wavs,frame_length,frame_step_in)
    a = tf.contrib.signal.inverse_stft(s,frame_length,frame_step_out)
    return tf.map_fn(tf_get_word,wavs,parallel_iterations=120,back_prop=False)

def tf_get_word(wav,size=16000):
    frames = shape_ops.frame(wav,size,300,pad_end=True)
    frame_stack = tf.stack(frames)
    frame_vols = tf.reduce_mean(tf.pow(frame_stack,2),axis=1)
    max_frame_vol = tf.argmax(frame_vols)
    return frame_stack[max_frame_vol,:]

def tf_pad(wav):
    """This NEEDS to be done better"""
    word_frame_size = tf.random_uniform([],12000,16000,dtype=tf.int32)
    left_pad = tf.random_uniform([],0,16000 - word_frame_size,dtype=tf.int32)
    right_pad = 16000 - word_frame_size - left_pad
    wav = tf_get_word(wav,word_frame_size)
    wav = tf.pad(wav,[[left_pad,right_pad]])
    return wav

def tf_simple_pad(wav):
    pad = tf.random_uniform([],-100,100,dtype=tf.int32)
    wav = tf.cond(tf.less(pad,0),
                  lambda: tf.slice(tf.pad(wav,[[0,-pad]]),[-pad],[16000]),
                  lambda: tf.slice(tf.pad(wav,[[pad,0]]),[0],[16000])
    )
    return wav

def tf_add_noise(wav,bg_wav):
    # I'll let NumPy handle all the picking of the WAV files, etc
    return wav + bg_wav


def tf_volume_equalize(wav,vary=False):
    if vary:
        control_vol = tf.truncated_normal([],0.1,0.01) # since the peak volume strategy I picked isn't perfect
    else:
        control_vol = tf.convert_to_tensor(0.1,tf.float32)
    chunks = tf.split(wav,50)
    vols = [tf.sqrt(tf.reduce_mean(tf.pow(chunk,2))) for chunk in chunks]
    vols = tf.stack(vols)
    max_vol = tf.reduce_max(vols)
    new_wav = tf.cond(tf.equal(max_vol,0),lambda: wav, lambda: tf.clip_by_value(wav*control_vol/max_vol,-1.0,1.0))
    return new_wav

def tf_resample(wav):
    resampled = tf.image.resize_images(tf.reshape(wav,[-1,1,1]),[16000,1])
    return tf.reshape(resampled,[resampled.shape[0]])

def tf_phase_vocode(s,frame_step_in,sampling_rate=16000):
    """This is unneccesary, even bad for some reason"""
    delta_t = tf.convert_to_tensor(frame_step_in / sampling_rate,tf.complex64)
    imag_i = tf.convert_to_tensor(1j,tf.complex64)
    print(imag_i.dtype)
    frames = tf.unstack(s)
    phase_shift = tf.zeros(s.shape[1],tf.complex64)
    for i, frame_tup in enumerate(zip(frames[:-1],frames[1:])):
        frame1, frame2 = frame_tup

        phase_change = tf.cast(tf.angle(frame2) - tf.angle(frame1),tf.complex64)

        freq_deviation = phase_change/delta_t - frame2
        freq_dev_angle = tf.mod(tf.angle(freq_deviation) + np.pi,2*np.pi) - np.pi
        freq_dev_angle = tf.cast(freq_dev_angle,tf.complex64)
        freq_dev_mag = tf.abs(freq_deviation)
        freq_dev_mag = tf.cast(freq_dev_mag,tf.complex64)
        wrapped_freq_deviation = freq_dev_mag * tf.exp(freq_dev_angle*imag_i)
        true_freq = frame2 + wrapped_freq_deviation

        phase_shift = phase_shift + delta_t*true_freq
        true_bins = tf.cast(tf.abs(frame2),tf.complex64)*tf.exp(tf.cast(tf.angle(phase_shift),tf.complex64)*imag_i)
        frames[i+1] = true_bins
    return tf.stack(frames)



def wanted_word(w,bg_data):
    w = pad(w)
    # w = pitch_shift(w)
    w = add_noise(w,bg_data)
    return w

def unknown_word(w,speakers,bg_data):
    # distortion_picker = np.random.uniform(0,1)
    # if distortion_picker < 0.3:
    #     speaker = np.random.randint(0,len(speakers["train"]))
    #     words = speakers["train"][speaker][1]
    #     chosen = np.random.choice(len(words),2)
    #     word1 = words[chosen[0]]["data"]
    #     word2 = words[chosen[1]]["data"]
    #     if distortion_picker < 0.2:
    #         w = combine(word1,word2)
    #     else:
    #         w = add(word1,word2)
    # elif distortion_picker < 0.4:
    #     w = reverse(w)
    # elif distortion_picker < 0.5:
    #     w = wraparound(w)
    w = pad(w)
    # w = pitch_shift(w)
    w = add_noise(w,bg_data)
    return w

def volume_equalizer(wav):
    """Makes it so that the noisiest part of a WAV is the same for all WAVs"""
    control_vol = 0.1

    chunks = np.array_split(wav,8)
    vol = np.array([np.sqrt(np.mean(chunk**2)) for chunk in chunks])
    max_vol = vol.max()

    if max_vol == 0:
        return wav
    wav = wav * control_vol/max_vol
    wav = np.clip(wav,-1.0,1.0)
    return wav


def get_word(wav,percent_wav=0.5,indices=False):
    chunk_size = 50
    keep_chunks = int(chunk_size*percent_wav)
    chunks = np.array_split(wav,50)
    chunk_samples = chunks[0].shape[0]
    dbs = [20*np.log10( np.sqrt(np.mean(chunk**2)) + 1e-8) for chunk in chunks]
    rolling_avg_db = np.array([np.mean(dbs[i:i+keep_chunks]) for i in range(0,chunk_size - keep_chunks + 1)])
    max_chunk_start = np.argmax(rolling_avg_db)
    if not indices:
        return np.concatenate(chunks[max_chunk_start:max_chunk_start+keep_chunks])
    else:
        return max_chunk_start*chunk_samples, (max_chunk_start + keep_chunks)*chunk_samples

def pad(wav):
    start_word, end_word = get_word(wav,0.8,indices=True)
    padding = np.random.randint(-start_word,sample_rate - end_word)
    if padding > 0:
        b = np.pad(wav,(padding,0),mode="constant")[:-padding]
    else:
        b = np.pad(wav,(0,-padding),mode="constant")[-padding:]
    return b

# def pad(d):
#     max_pad = 100
#     pad_num = np.random.randint(-max_pad,max_pad)
#     if pad_num > 0:
#         b = np.pad(d,(pad_num,0),mode="constant")[:-pad_num]
#     else:
#         b = np.pad(d,(0,-pad_num),mode="constant")[-pad_num:]
#     return b

def white_noise():
    return (np.clip(np.random.randn(16000,)*0.1,-1,1)).astype(np.float32)

def red_noise(white_noise,r=0.5):
    """0 < r < 1, zero is white noise, bigger r shifts more towards lower frequencies"""
    white = white_noise
    red = np.zeros((white_noise.shape[0]),dtype=np.float32)
    red[0] = white[0]
    for i,ele in enumerate(white[:-1]):
        red[i+1] = r*red[i] + ((1-r**2)**0.5)*white[i+1]
    return red

# Further improvements:
# use the white noise (random uniform) and pink and blue and violet noises provided in the kernel (on your Jupyter now)
# Multiply slices together too
# maybe use combos of 10 of the training samples, all very quiet, to simulate real background conversation (no, sounds like words still)
# adjust optimal background volume
# also try adding effects like reverb, echo, flange, phase, etc to words
# def get_noise(bg_data):
#     background_frequency = 0.8
#     max_background_volume = 0.15
#     bg_sounds = []
#     for _ in range(2):
#         if np.random.uniform(0,1) < 0.5: # use regular background noise
#             bg_index = np.random.randint(len(bg_data))
#             bg_samp = bg_data[bg_index]
#             bg_offset = np.random.randint(0,len(bg_samp) - sample_rate)
#             bg_sliced = bg_samp[bg_offset:(bg_offset + sample_rate)]
#         else: # use generated white and red noise
#             if np.random.uniform(0,1) < 0.25:
#                 bg_sliced = white_noise()
#             else:
#                 r = np.random.uniform(0.01,0.99)
#                 bg_sliced = red_noise(r)
#         bg_sliced = np.clip(bg_sliced*0.1/np.sqrt(np.mean(bg_sliced**2)),-1.0,1.0)
#         bg_sounds.append(bg_sliced)
#     combiner = np.random.uniform(0,1)
#     if combiner < 0.66:
#         # can try this if model is still not generalizing
#         sound1_ratio = np.random.uniform(0,1)
#         sound2_ratio = 1 - sound1_ratio
#         bg_combined = sound1_ratio*bg_sounds[0] + sound2_ratio*bg_sounds[1]
#         # bg_combined = bg_sounds[0] + bg_sounds[1]
#     else:
#         bg_combined = bg_sounds[0]
#     bg_combined = np.clip(bg_combined*0.1/np.sqrt(np.mean(bg_combined**2)),-1.0,1.0)
#     if np.random.uniform(0,1) < background_frequency:
#         bg_volume = np.random.uniform(0,max_background_volume)
#     else:
#         bg_volume = 0
#     return bg_volume*bg_combined

def get_noise(bg_data,val=False):
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
    return bg_volume*bg_sliced

# augmentation_noises = []
# for color in ["white","blue","violet","pink","brown"]:
#     augmentation_noises.append(acoustics.generator.noise(16000*1000,color).astype(np.float32))
# for r in [.25,.5,.75]:
#     augmentation_noises.append(red_noise(augmentation_noises[0],r))

# def get_noise(bg_data,val=False):
#     """This method uses the same SNR ratio distribution (after equalizing volumes) as the original method, and should just have better results with the augmented noise"""
#     if not val: # don't validate the silence labels based on my made up noise, since it might be easier to classify and give me false hope
#         bg_data += augmentation_noises
#     control_volume = 0.1
#     background_frequency = 0.8
#     if np.random.uniform(0,1) > background_frequency:
#         bg_volume = 0
#     else:
#         # this will give NSRs (inverse of SNR) of truncated exponentially distributed around mean 0.03
#         exp_mean = 30
#         cdf = np.random.random()*0.99
#         bg_volume = -np.log(1 - cdf)/exp_mean
#     bg_index = np.random.randint(len(bg_data))
#     bg_samp = bg_data[bg_index]
#     bg_offset = np.random.randint(0,len(bg_samp) - sample_rate)
#     bg_sliced = bg_samp[bg_offset:(bg_offset + sample_rate)]
#     bg_sliced = bg_sliced*bg_volume*control_volume/np.sqrt(np.mean(bg_sliced**2))
#     return np.clip(bg_sliced,-1.0,1.0)


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
    d1 = np.pad(get_word(d1,0.5),(0,8000),mode="constant")
    d2 = np.pad(get_word(d2,0.5),(8000,0),mode="constant")
    return (d2*sb + d1*(1 - sb)).astype(np.float32)

def resample(a,samples=16000):
    resampler = sp.interpolate.interp1d(np.arange(a.shape[0]),a,kind='linear')
    return resampler((np.linspace(0,a.shape[0]-1,samples,endpoint=False))).astype(np.float32)

def speedx(a,f=None):
    if f is None:
        speed_picker = np.random.uniform(0,1)
        if speed_picker > 0.5:
            speed = np.random.uniform(0.85,1.15)
        else:
            return a
    else:
        speed = f
    samples = int(sample_rate / speed)
    a = resample(a,samples)
    sample_diff = np.abs(samples - sample_rate)
    left_samples = np.random.randint(0,sample_diff)
    right_samples = sample_diff - left_samples
    if speed > 1:
        return np.pad(a,(left_samples,right_samples),mode='constant')
    else:
        return a[left_samples:-right_samples]

def wraparound(a):
    cut = np.random.randint(0,a.shape[0])
    return np.append(a[cut:],a[:cut])

def pitch_shift(a):
    if np.random.uniform(0,1) < 0.2:
        pitch = np.random.uniform(-2,2)
    else:
        return a
    sampling_rate = 16000
    chunk = 300
    overlap = 0.75
    hop_in = int((1-overlap)*chunk)
    resample_x = 2**(pitch/12)
    hop_out = int(hop_in*resample_x)

    def stft(x):
        h = sp.hanning(chunk)
        X = np.array([np.fft.fft(h*x[i:i+chunk]) for i in range(0, len(x)-chunk, hop_in)])
        return X

    def istft(X):
        h = sp.hanning(chunk)
        x = np.zeros(len(X) * (hop_out))
        for n, i in enumerate(range(0, len(x)-chunk, hop_out)):
            x[i:i+chunk] += h*np.real(np.fft.ifft(X[n]))
        return x

    def vocode(frame1,frame2,phase_shift):
        delta_t = hop_in / sampling_rate

        phase_change = np.angle(frame2) - np.angle(frame1)

        freq_deviation = (phase_change)/delta_t - frame2
        freq_dev_angle = np.mod(np.angle(freq_deviation) + np.pi,2*np.pi) - np.pi
        freq_dev_mag = np.abs(freq_deviation)
        wrapped_freq_deviation = freq_dev_mag * np.exp(freq_dev_angle*1j)
        true_freq = frame2 + wrapped_freq_deviation # this is the true frequency given that things fall bt bins

        phase_shift += delta_t * true_freq
        true_bins = np.abs(frame2)*np.exp(np.angle(phase_shift)*1j)
        return true_bins,phase_shift


    s = stft(a)
    s_vocoded = np.zeros(s.shape)
    phase_shift = np.zeros(s.shape[1],dtype=np.complex128)
    s_vocoded[0,:] = s[0,:]
    for i,frames in enumerate(zip(s[:-1],s[1:])):
        s_vocoded[i+1,:], phase_shift = vocode(frames[0],frames[1],phase_shift)

    a_shifted = istft(s).astype(np.float32)
    return resample(a_shifted,sample_rate)
