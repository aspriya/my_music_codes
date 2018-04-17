import pyaudio
import numpy as np
from scipy.signal import argrelextrema
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


##Some settings

FORMAT = pyaudio.paFloat32
CHANNELS = 1
FS = 44100
CHUNK = 256
NFFT = 2048
OVERLAP = 0.5
PLOTSIZE = 32*CHUNK
N = 4

freq_range = np.linspace(10, FS/2, NFFT//2 + 1)
df = FS/NFFT
HOP = NFFT*(1-OVERLAP)


##Some preliminary functions

def db_spectrum(data) : #computes positive frequency power spectrum
    fft_input = data*np.hanning(NFFT)
    spectrum = abs(np.fft.rfft(fft_input))/NFFT
    spectrum[1:-1] *= 2
    return 20*np.log10(spectrum)

def highest_peaks(spectrum) : #finds peaks (local maxima) and picks the N highest ones
    peak_indices = argrelextrema(spectrum, np.greater)[0]
    peak_values = spectrum[peak_indices]
    highest_peak_indices = np.argpartition(peak_values, -N)[-N:]
    return peak_indices[(highest_peak_indices)]

def detection_plot(peaks) : #formats data for vertical line plotting
    x = []
    y = []
    for peak in peaks :
        x.append(peak*df)
        x.append(peak*df)
        y.append(-200)
        y.append(0)
    return x, y


##Main class containing loop and UI

class SpectrumAnalyzer(pg.GraphicsWindow) :

    def __init__(self) :
        super().__init__() 
        self.initUI()
        self.initTimer()
        self.initData()
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = FORMAT,
                                   channels = CHANNELS,
                                   rate = FS,
                                   input = True,
                                   output = True,
                                   frames_per_buffer = CHUNK)

    def initUI(self) :
        self.setWindowTitle("Microphone Audio Data")
        audio_plot = self.addPlot(title="Waveform")
        audio_plot.showGrid(True, True)
        audio_plot.addLegend()
        audio_plot.setYRange(-1,1)
        self.time_curve = audio_plot.plot()
        self.nextRow()
        fft_plot = self.addPlot(title="FFT") 
        fft_plot.showGrid(True, True)
        fft_plot.addLegend()
        fft_plot.setLogMode(True, False)
        fft_plot.setYRange(-140,0) #may be adjusted depending on your input
        self.fft_curve = fft_plot.plot(pen='y')
        self.detection = fft_plot.plot(pen='r')

    def initTimer(self) :
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def initData(self) :
        self.waveform_data = np.zeros(PLOTSIZE)
        self.fft_data = np.zeros(NFFT)
        self.fft_counter = 0

    def closeEvent(self, event) :
        self.timer.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def update(self) :

        raw_data = self.stream.read(CHUNK)
        self.stream.write(raw_data, CHUNK)
        self.fft_counter += CHUNK

        sample_data = np.fromstring(raw_data, dtype=np.float32)
        self.waveform_data = np.concatenate([self.waveform_data, sample_data])  #update plot data
        self.waveform_data = self.waveform_data[CHUNK:]                         #
        self.time_curve.setData(self.waveform_data)

        self.fft_data = np.concatenate([self.fft_data, sample_data])    #update fft input
        self.fft_data = self.fft_data[CHUNK:]                           #
        if self.fft_counter == HOP :
            spectrum = db_spectrum(self.fft_data)
            peaks = highest_peaks(spectrum)
            x, y = detection_plot(peaks)
            self.detection.setData(x, y, connect = 'pairs')
            self.fft_curve.setData(freq_range, spectrum)
            self.fft_counter = 0

if __name__ == '__main__':
    spec = SpectrumAnalyzer()