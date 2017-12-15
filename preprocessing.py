import tensorflow as tf
import numpy as np
from toolz.functoolz import memoize
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

window_size_samples = 30 * 16 # ms * samples/ms
window_stride_samples = 10 * 16


def preprocess(wavs):
    """Steps to add background noise, silence, get time offsets, etc."""
    def distort_wav(ph_tup):
        wav_ph, volume_ph, time_shift_padding_ph, time_shift_offset_ph, bg_ph, bg_volume_ph = ph_tup
        wav = tf.reshape(wav_ph,[sample_rate,1])
        scaled_wav = tf.multiply(wav, volume_ph)
        # maybe i should do some rescaling here for really quiet waves
        padded_wav = tf.pad(scaled_wav,time_shift_padding_ph,mode="CONSTANT")
        sliced_wav = tf.slice(padded_wav,time_shift_offset_ph,[sample_rate,-1])
        scaled_bg = tf.multiply(bg_ph,bg_volume_ph)
        wav_with_bg = tf.add(sliced_wav,scaled_bg)
        clamped_wav = tf.clip_by_value(wav_with_bg,-1.0,1.0)
        return clamped_wav

    distorted_wav = tf.map_fn(
        distort_wav,
        [wavs, volume_ph, time_shift_padding_ph, time_shift_offset_ph, bg_ph, bg_volume_ph],
        dtype=tf.float32,
        parallel_iterations=100,
        back_prop=False
    )
    return distorted_wav


# Different features from preprocessing
# this will generate 64 features
def make_log_mel_fb(sig,name=None):
    with tf.name_scope(name,"audio_processing",[sig]) as scope:
        stfts = tf.contrib.signal.stft(sig, frame_length=window_size_samples, frame_step=window_stride_samples,fft_length=1024)
        magnitude_spectrograms = tf.abs(stfts)
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 98
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
          upper_edge_hertz)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        # Note: Shape inference for `tf.tensordot` does not currently handle this case.
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
        return log_mel_spectrograms

def make_mfccs(sig):
    def make_1_mfcc(s):
        spectrogram = contrib_audio.audio_spectrogram(
            s,
            window_size = window_size_samples,
            stride = window_stride_samples,
            magnitude_squared = True
        )
        mfcc = contrib_audio.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count = 40
        )
        mfcc_2d = tf.reshape(mfcc,[mfcc.shape[1],mfcc.shape[2]])
        return mfcc_2d

    return tf.map_fn(make_1_mfcc,sig,parallel_iterations=100,back_prop=False,dtype=tf.float32)


def pad(d):
    max_pad = 10
    pad_num = np.random.randint(-max_pad,max_pad)
    if pad_num > 0:
        b = np.pad(d,(pad_num,0),mode="constant")[:-pad_num]
    else:
        b = np.pad(d,(0,-pad_num),mode="constant")[-pad_num:]
    return d

def add_noise(d,bg_data):
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
