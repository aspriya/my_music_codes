import pyaudio
import numpy as np

def record(rate=44100, chunk_size=1024, record_seconds=10):
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaduio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)
	frames = [] # A python-list of chunks(numpy.ndarray)
	for _ in range(0, int(rate / chunk_size * record_seconds)):
	    data = stream.read(CHUNKSIZE)
	    frames.append(np.fromstring(data, dtype=np.int16))

	#Convert the list of numpy-arrays into a 1D array (column-wise)
	numpydata = np.hstack(frames)

	# close stream
	stream.stop_stream()
	stream.close()
	p.terminate()

	plt.plot(frames)
	plt.show()
	return frames

frames = record()