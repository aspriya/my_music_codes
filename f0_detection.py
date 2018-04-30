import numpy as np
from scipy.signal import get_window
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'f0_detect_files/'))
from scipy.io import wavfile #to read and write wavfiles

import dftModel as DFT
import stft as STFT
import harmonicModel as HM
import sineModel as SM
from matplotlib import pyplot as plt

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}


def detect_f0(audio_path, window_size, Hop_size):

    fs, data = wavfile.read(audio_path)
    data = np.float32(data)/norm_fact[data.dtype.name]
    window_length_in_samples = window_size

    length_of_audio = len(data) / float(fs)

    w = get_window('hanning', window_length_in_samples)
    N = 2048 * 2
    t = -50
    minf0 = 100
    maxf0 = 700
    f0et = 7
    H = Hop_size

    f0 = HM.f0Detection(data, fs, w, N, H, t, minf0, maxf0, f0et)
    return f0
    # print("length of f0 array : ", len(f0))
    # print("and number of windows: ", len(data)/window_length_in_samples, " *  Number of Hops per window : ", N/H," is = ",  (N/H) * (len(data)/window_length_in_samples)  )
    # plt.plot(f0);
    # plt.show()
