import numpy as np


def short_term_feauture_extraction(signal, fs, window_len, step_len):
	'''
	This function implements a shot-term windowing process. For each window, a set of features are extracted. 
	The rusult is a sequence of feature vectros, stored in a numpy matrix.	
	
	ARGUMENTS
		signal:				the input singnal in samples
		fs:						sampling rate of the signal
		window_len:		length of short term window in samples
		step_len:			length of step size in samples

	RETURNS
		short_term_features:	a numpy array(matrix) of shape: (num_of_features x num_of_short_term_windows)
	'''

	window_len = int(window_len)
	step_len = int(step_len) 

	signal = np.double(signal) #normalize the signal

	
	'''
	In a wav file, one sample is stored in two bytes. so each sample is 16 bit long.
	In WAV, 16-bit is signed and little-endian. So if you take the value of that 16-bit 
	sample and divide it by 2^15, you'll end up with a sample that is normalized to be 
	within the range -1 to 1.
	'''
	signal = signal / (2 ** 15) #normalizing the sample to be in -1 to 1 range in value.
	DC = signal.mean()
	MAX = (np.abs(signal)).max()
	signal = (signal - DC) / (MAX + 0.0000000001)

	N = len(signal)
	current_position = 0  #current position in samples
	current_frame = 0
	nFFT = window_len / 2 

	num_of_features = 3
	short_term_features = []
	
	while(current_position + window_len - 1 < N):  #loop each st_window until the end of signal
		current_frame += 1
		x = signal[current_position : current_position+window_len] #get the current window
		current_position = current_position + step_len #update current position

		current_feature_vector = np.zeros((num_of_features, 1))
		current_feature_vector[0] = short_term_ZCR(x)  #zero crossing rate
		current_feature_vector[1] = short_term_Energy(x)
		current_feature_vector[2] = short_term_Energy_Entropy(x)

		short_term_features.append(current_feature_vector)

	short_term_features = np.concatenate(short_term_features, 1)
	# Concatenation refers to joining. This function is used to join two or more arrays of the same shape 
	# along a specified axis. here the specified axis is 1.
	print("feature ext number of frames :" + str(current_frame))
	return short_term_features


eps = 0.00000001

""" Time-domain audio feature extraction functions """

def short_term_ZCR(frame):
  """Computes zero crossing rate of frame"""
  count = len(frame)
  countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
  return (np.float64(countZ) / np.float64(count-1.0))


def short_term_Energy(frame):
  """Computes signal energy of frame"""
  return np.sum(frame ** 2) / np.float64(len(frame))


def short_term_Energy_Entropy(frame, numOfShortBlocks=10):
  """Computes entropy of energy"""
  Eol = np.sum(frame ** 2)    # total frame energy
  L = len(frame)
  subWinLength = int(np.floor(L / numOfShortBlocks))
  if (L != subWinLength * numOfShortBlocks):
  	frame = frame[0:subWinLength * numOfShortBlocks]

  # subWindows is of size [numOfShortBlocks x L]
	subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

  # Compute normalized sub-frame energies
	s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)

  # Compute entropy of the normalized sub-frame energies:
	Entropy = -np.sum(s * np.log2(s + eps))
  return Entropy