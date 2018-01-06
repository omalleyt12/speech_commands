import tensorflow as tf
import numpy as np
from toolz.functoolz import memoize
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import mel_matrix

sample_rate = 16000
window_size_samples = 30 * 16 # ms * samples/ms
window_stride_samples = 10 * 16


def make_features(wavs,is_training,name="log-mel"):
    if name == "log-mel":
        print("Features: Log Mel")
        return make_vtlp_mels(wavs,is_training,bins=128)
    elif name == "log-mel-40":
        return make_vtlp_mels(wavs,is_training,bins=40)
    elif name == "log-mel-40-energy":
        return make_vtlp_mels(wavs,is_training,bins=40,energies=True)
    elif name == "log-mel-energy":
        return make_vtlp_mels(wavs,is_training,bins=128,energies=True)
    elif name == "mfcc":
        print("Features: MFCC")
        return make_mfccs(wavs)
    else:
        print("Features: Identity")
        return wavs

# Different features from preprocessing
# this will generate 64 features
def make_log_mel_fb(sig,name=None):
    with tf.name_scope(name,"audio_processing",[sig]) as scope:
        stfts = tf.contrib.signal.stft(sig, frame_length=window_size_samples, frame_step=window_stride_samples,fft_length=1024)
        magnitude_spectrograms = tf.abs(stfts)
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 7600.0, 128
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
          upper_edge_hertz)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        # Note: Shape inference for `tf.tensordot` does not currently handle this case.
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
        return log_mel_spectrograms

def make_vtlp_mels(sig,is_training,name=None,bins=128,frame_stride_ms=10):
    """A limitation with this approach is that the VTLP factor is the same within a batch, but individual VTLP did NOT help, so there's that"""
    window_stride_samples = frame_stride_ms*16
    with tf.name_scope(name,"audio_processing",[sig]) as scope:
        stfts = tf.contrib.signal.stft(sig, frame_length=window_size_samples, frame_step=window_stride_samples,fft_length=1024)
        magnitude_spectrograms = tf.abs(stfts)
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, bins
        # still keep this here for shape inference I guess
        linear_to_mel_weight_matrix = mel_matrix.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz,is_training)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        # Note: Shape inference for `tf.tensordot` does not currently handle this case.
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

        # scale the 128 bins log-mels to have zero mean and unit var
        # wow, this really made things WORSE somehow
        # if bins == 128:
        #     log_mel_spectrograms = (log_mel_spectrograms + 2.76295)/3.19395

        # if energies:
        #     frame_energies = tf.sqrt(tf.reduce_mean(magnitude_spectrograms**2,axis=2,keep_dims=True))
        #     # subtract the average frame energy
        #     frame_energies = frame_energies - tf.reduce_mean(frame_energies,axis=1,keep_dims=True)
        #     return tf.concat([frame_energies,log_mel_spectrograms],axis=2)
        return log_mel_spectrograms

def make_mfccs(sig):
    def make_1_mfcc(s):
        spectrogram = contrib_audio.audio_spectrogram(
            tf.reshape(s,(s.shape[0],1)),
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

