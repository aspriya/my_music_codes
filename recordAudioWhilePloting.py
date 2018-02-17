import pyaudio
import numpy as np
from progressbar import ProgressBar
import matplotlib.pyplot as plt


prgrsBar = ProgressBar()

def record(rate=44100, chunk_size=1024, record_seconds=3):
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)
	frames = [] # A python-list of chunks(numpy.ndarray)
	for _ in prgrsBar(range(0, int((rate * record_seconds) / chunk_size ))):  #samples are read as chunks from the stream
	    data = stream.read(chunk_size)
	    frames.append(np.fromstring(data, dtype=np.int16))

	#Convert the list of numpy-arrays into a 1D array (column-wise)
	numpydata = np.hstack(frames)

	# close stream
	stream.stop_stream()
	stream.close()
	p.terminate()
	
	return numpydata