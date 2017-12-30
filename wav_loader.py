import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

sample_rate = 16000


class WavLoader:
    def __init__(self,name=None,desired_samples=None):
        with tf.name_scope("wav_loader",name) as scope:
            self.wav_filename_ph = tf.placeholder(tf.string,[])
            self.wav_loader = io_ops.read_file(self.wav_filename_ph)
            if desired_samples is None:
                self.wav_decoder = contrib_audio.decode_wav(self.wav_loader,desired_channels=1)
            else:
                self.wav_decoder = contrib_audio.decode_wav(self.wav_loader,desired_channels=1,desired_samples=desired_samples)

    def load(self,f,sess):
        return sess.run(self.wav_decoder,{self.wav_filename_ph:f}).audio.flatten()

def load_test_data(sess):
    wav_loader = WavLoader("test_wav_loader",desired_samples=sample_rate)
    test_dir = "test/audio"
    test_index = []
    for i,wav_path in enumerate(gfile.Glob("test/audio/*.wav")):
        if i % 10000 == 0: print("Test {}".format(i))
        tdata = wav_loader.load(wav_path,sess)
        file_name = os.path.basename(wav_path)
        test_index.append({"file":wav_path,"identifier":file_name,"data":tdata})
    return test_index


def load_bg_data(sess):
    wav_loader = WavLoader("bg_wav_loader")
    bg_data = []
    bg_path = "train/audio/_background_noise_/*.wav"
    for wav_path in gfile.Glob(bg_path):
        wav_data = wav_loader.load(wav_path,sess)
        bg_data.append(wav_data)
    return bg_data


