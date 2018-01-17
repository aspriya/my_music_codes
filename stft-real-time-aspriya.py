from scipy import signal
from scipy.io import wavfile #to read and write wavfiles
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import style

style.use('ggplot')

fs, data = wavfile.read('piano.wav')
length_of_audio = len(data) / fs
window = 'hanning'

window_length_in_milliseconds = 20
window_length_in_samples = (fs *window_length_in_milliseconds) / 1000
nperseg = window_length_in_samples
nfft = 1024 #lenght of fft window

print("window size is " + str(window_length_in_samples))
print("sample rate is " + str(fs))
print("time of audio is "+ str(length_of_audio))

f, t, Zxx = signal.stft(data, fs, nperseg=nperseg, nfft=nfft, window=window)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.axis([0, length_of_audio, 0, 6000])
plt.title('STFT Magnitude')
plt.ylabel('Frequency in Hz')
plt.xlabel('Time in sec')

print("Shape of Zxx, which is a matrix. rows are frames, and columns are fft coificiants" + str(Zxx.shape))
plt.show()