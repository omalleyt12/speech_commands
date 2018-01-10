import tensorflow as tf
import numpy as np
from toolz.functoolz import memoize
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import mel_matrix

sample_rate = 16000
window_size_samples = 30 * 16 # ms * samples/ms
window_stride_samples = 10 * 16


def make_features(wavs,is_training,name="log-mel"):
    if name == "log-mel":
        return make_vtlp_mels(wavs,is_training,bins=120)
    elif name == "equal-log-mel":
        return make_vtlp_mels(wavs,is_training,bins=120,frame_equalize=True)
    elif name == "multi-mels":
        return multi_mels(wavs,is_training,stride_ms=10)
    elif name == "mega-multi-mels":
        return multi_mels(wavs,is_training,bins=256,stride_ms=5)
    elif name == "log-mel-40":
        return make_vtlp_mels(wavs,is_training,bins=40)
    elif name == "mfcc":
        return make_vtlp_mfccs(wavs,is_training)
    elif name == "mfcc-13":
        return make_vtlp_mfccs(wavs,is_training,num_mfccs=13)
    elif name == "mel":
        return make_vtlp_mels(wavs,is_training,bins=120,log=False)
    else:
        print("Features: Identity")
        return wavs

# Different features from preprocessing
def make_log_mel_fb(sig,name=None):
    with tf.name_scope(name,"audio_processing",[sig]) as scope:
        stfts = tf.contrib.signal.stft(sig, frame_length=window_size_samples, frame_step=window_stride_samples)
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

def make_vtlp_mels(sig,is_training,name=None,bins=128,log=True,frame_equalize=False):
    """A limitation with this approach is that the VTLP factor is the same within a batch, but individual VTLP did NOT help, so there's that"""
    print("Using {} Log-Mels".format(bins))
    with tf.name_scope(name,"audio_processing",[sig]) as scope:
        vtlp = tf.cond(is_training,lambda: tf.truncated_normal([],1.0,0.1,dtype=tf.float64), lambda: tf.constant(1.0,tf.float64))
        stfts = tf.contrib.signal.stft(sig, frame_length=window_size_samples, frame_step=window_stride_samples,pad_end=True)
        magnitude_spectrograms = tf.abs(stfts)
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, bins
        # still keep this here for shape inference I guess
        linear_to_mel_weight_matrix = mel_matrix.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz,vtlp)
        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        # Note: Shape inference for `tf.tensordot` does not currently handle this case.
        mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        if frame_equalize:
            print("Frame equalizing")
            # every frame will have the same volume, might help for learning features if volumes don't matter too much
            mel_spectrograms = mel_spectrograms*200.0/(tf.reduce_sum(mel_spectrograms,axis=2,keep_dims=True) + 1e-6)

        if not log:
            return mel_spectrograms

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

def make_vtlp_mfccs(sig,is_training,name=None,num_mfccs=13):
    """This will be used to mirror Hello Edge or Heng's architectures on MFCC"""
    print("Using MFCCs")
    log_mels = make_vtlp_mels(sig,is_training,bins=40)
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mels)[:num_mfccs]
    return mfccs

def multi_mels(sig,is_training,name=None,bins=120,stride_ms=5):
    with tf.name_scope(name,"multi-mels",[sig]) as scope:
        # ensure each spectrogram is VTLP'd the same way
        vtlp = tf.cond(is_training,lambda: tf.truncated_normal([],1.0,0.1,dtype=tf.float64), lambda: tf.constant(1.0,tf.float64))
        window_stride_samples = stride_ms * 16
        mels_list = []
        for window_size_ms in [8,16,32,64]:
            window_size_samples = window_size_ms * 16
            # align the STFTs so that each one starts with its center at the beginning of the WAV and ends with its center at the end of the wave
            pad_ends = int(window_size_samples / 2)
            print(sig.shape)
            padded_sig = tf.pad(sig,[[0,0],[pad_ends,pad_ends]])
            stfts = tf.contrib.signal.stft(padded_sig, frame_length=window_size_samples, frame_step=window_stride_samples)
            magnitude_spectrograms = tf.abs(stfts)
            # Warp the linear-scale, magnitude spectrograms into the mel-scale.
            num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
            lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, bins
            # still keep this here for shape inference I guess
            linear_to_mel_weight_matrix = mel_matrix.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                upper_edge_hertz,vtlp)
            mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
            log_offset = 1e-6
            log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
            mels_list.append(log_mel_spectrograms)
        stacked_log_mels = tf.stack(mels_list,axis=3)
        return stacked_log_mels

# def make_mfccs(sig):
#     def make_1_mfcc(s):
#         spectrogram = contrib_audio.audio_spectrogram(
#             tf.reshape(s,(s.shape[0],1)),
#             window_size = window_size_samples,
#             stride = window_stride_samples,
#             magnitude_squared = True
#         )
#         mfcc = contrib_audio.mfcc(
#             spectrogram,
#             sample_rate,
#             dct_coefficient_count = 40
#         )
#         mfcc_2d = tf.reshape(mfcc,[mfcc.shape[1],mfcc.shape[2]])
#         return mfcc_2d

#     return tf.map_fn(make_1_mfcc,sig,parallel_iterations=100,back_prop=False,dtype=tf.float32)

