import numpy as np
import scipy as sp
from toolz.functoolz import memoize

sample_rate = 16000

def wanted_word(w,bg_data):
    w = pad(w)
    w = pitch_shift(w)
    w = add_noise(w,bg_data)
    return w

def unknown_word(w,speakers,bg_data):
    distortion_picker = np.random.uniform(0,1)
    if distortion_picker < 0.3:
        speaker = np.random.randint(0,len(speakers["train"]))
        words = speakers["train"][speaker][1]
        chosen = np.random.choice(len(words),2)
        word1 = words[chosen[0]]["data"]
        word2 = words[chosen[1]]["data"]
        if distortion_picker < 0.2:
            w = combine(word1,word2)
        else:
            w = add(word1,word2)
    elif distortion_picker < 0.4:
        w = reverse(w)
    elif distortion_picker < 0.5:
        w = wraparound(w)
    w = pad(w)
    w = pitch_shift(w)
    w = add_noise(w,bg_data)
    return w


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
