import tensorflow as tf
import numpy as np
from toolz.functoolz import memoize
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

window_size_samples = 30 * 16 # ms * samples/ms
window_stride_samples = 10 * 16


def make_features(wavs,name="log-mel"):
    if name == "log-mel":
        return make_log_mel_fb(wavs)
    else:
        return make_mfccs(wavs)

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

