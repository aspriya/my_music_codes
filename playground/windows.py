import numpy as np 
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt

# window size, change if you need a different size window.
# window size means the number of samples in the window
M = 63

#change here to 'hamming or blackman or blackmanharris or hanning to get those windows'
window = get_window('hanning', M)
hM1 = int(math.floor((M+1)/2)) # 63+1 / 2 = 32
hM2 = int(math.floor(M/2))  # 63 / 2 = 31

N = 512 #number of samples to take fft should be power of 2
hN = N/2  #256
fftbuffer = np.zeros(N) #make a array of size N with zeros filled
fftbuffer[:hM1] = window[hM2:] #put second part of window samples as first part of fft buffer
fftbuffer[N-hM2:] = window[:hM2] #put first part of window samples at last in fft buffer

X = fft(fftbuffer) #calculate the fft of signal which is in fft buffer
absX = abs(X)  # calculating the amplitude of bins. 

# if values are lower than that of minimum float value that python float can represent, make them
# to minum value of python float
absX[absX<np.finfo(float).eps] = np.finfo(float).eps

mX = 20*np.log10(absX) # calculate the amplitude as decibels
pX = np.angle(X)

mX1 = np.zeros(N)
mX1[:hN] = mX[hN:] 
mX1[N-hN:] = mX[:hN]

plt.plot(np.arange(-hN, hN)/float(N)*M, mX1-max(mX1)) #two axis
plt.axis([-20, 20, -80, 0]) #display only the values within range of in x axis: -20 to 20, y axis: -80 to 0
plt.show()