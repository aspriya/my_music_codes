from scipy import signal
from scipy.io import wavfile #to read and write wavfiles
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import style
from featureExtractionAspriya import short_term_feauture_extraction
from spectralFeatureExtractionAspriya import stSpectogram
from recordAudioWhilePloting import record

# style.use('ggplot')

#defining a GridSpec for plots with 5 rows and 1 column
fig = plt.figure(figsize=(8,6))
grid = plt.GridSpec(7,1, hspace=2)

#if importing a wav file use following


def record():
    fs = 44100
    data = record(rate=fs, chunk_size=1024, record_seconds=10)
    return fs, data

def generate_timeArray():
    # gives a array containing numbers from 1 to number of samples
    timeArray = np.arange(len(data));
    timeArray = timeArray / float(fs)   #convert to seconds
    # timeArray = timeArray * 1000 #scale to milliseconds
    return timeArray

fs, data = wavfile.read('onsetDetectionData/sounds/25-rujero.wav')
length_of_audio = len(data) / float(fs)
window = 'hanning'

#time signal graph
time_signal_plot = fig.add_subplot(grid[0, :])
timeArray = generate_timeArray()
time_signal_plot.plot(timeArray, data)

#window_length_in_milliseconds = 92
#window_length_in_samples = (fs *window_length_in_milliseconds) / 1000
window_length_in_samples = 2048 #1024,2048,4096,8192
nperseg = window_length_in_samples
nfft = None #lenght of fft window. if zeor padding is neades, use a suitable power of 2 value here

print("window size is " + str(window_length_in_samples))
print("sample rate is " + str(fs))
print("time of audio is "+ str(length_of_audio))
print("number of sampels is "+ str(len(data)))
print("number of windows is "+ str(len(data)/window_length_in_samples))


step_size_in_samples = window_length_in_samples//4
print("step size in sampels is ", step_size_in_samples)

short_term_feautures = short_term_feauture_extraction(data, fs, window_length_in_samples, step_size_in_samples)
print ("feature matrix shape is: " +str(short_term_feautures.shape))

# energy plot
energy_graph = fig.add_subplot(grid[1, :])
energy_graph.plot(short_term_feautures[1])

# # zero crossing rate plot
# zero_cross_rate_graph = fig.add_subplot(grid[2, :])
# zero_cross_rate_graph.plot(short_term_feautures[0])

# enrgy entropy plot
energy_entropy_graph = fig.add_subplot(grid[2, :])
energy_entropy_graph.plot(short_term_feautures[2])


# calculate the spectrogram
specgram, TimeAxis, FreqAxis = stSpectogram(data, fs, window_length_in_samples, step_size_in_samples, True)
print("spectrogram is: ",  specgram)
print("spectogram length ", len(specgram))
print("spectrogram shape ", specgram.shape)


#........................... PREPARING ONSET ARRAY (WETHER A ONSET DETECTED OR NOT AT THAT TIME INTERVAL) ...........................
# Length of onset array should be number of frames - 5 (since the step size is "window_length_in_samples / 4")
crnt_fram = 1
onsetArray = []
window_length_in_seconds = window_length_in_samples / float(fs)
step_size_in_seconds = step_size_in_samples / float(fs)
print(" ")
print("-----------------------------------------------------------")
print("window length in seconds : ", window_length_in_seconds)
print("step size in seconds : ", step_size_in_seconds)

while (crnt_fram <= len(short_term_feautures[0]) - 5):
    t1 = window_length_in_seconds + ((crnt_fram - 1) * step_size_in_seconds )
    t2 = t1 + step_size_in_seconds
    crnt_fram = crnt_fram + 1

    #if t is in between t1 and t2, push 1 to onsetArray, else push 0.





# #calculate STFT
# f, t, Zxx = signal.stft(data, fs, nperseg=2048, noverlap=1536, nfft=nfft, window=window)
#
#
# #spectrogram graph
# spectrogram = fig.add_subplot(grid[3:, :])
# spectrogram.pcolormesh(t, f, np.abs(Zxx))
# spectrogram.axis([0, length_of_audio, 0, 2100])
# # spectrogram.set_title('STFT Magnitude')
# spectrogram.set_ylabel('Frequency in Hz')
# spectrogram.set_xlabel('Time in sec')
#
#
#
# print("Shape of Zxx, which is a matrix. rows are frames, and columns are fft coificiants" + str(Zxx.shape))
# plt.show()
