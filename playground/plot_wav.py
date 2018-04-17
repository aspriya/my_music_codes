from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

fs, data = wavfile.read('onsetDetectionData/sounds/25-rujero.wav');

# gives a array containing numbers from 1 to number of samples
timeArray = np.arange(len(data));
timeArray = timeArray / float(fs)   #convert to seconds
timeArray = timeArray * 1000 #scale to milliseconds
print(timeArray)

print("number of sampels: ", len(data))
print("sample rate: ", fs)
print("length of singal in milliseconds: ", timeArray[-1])

plt.plot(timeArray, data)
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.show()
