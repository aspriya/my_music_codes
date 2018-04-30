import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def short_term_feauture_extraction(signal, fs, window_len, step_len, PLOT=False):
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
	nfft = window_len / 2

	specgram = np.array([], dtype=np.float64)

	num_of_features = 7
	short_term_features = []

	while(current_position + window_len - 1 < N):  #loop each st_window until the end of signal
		current_frame += 1
		x = signal[current_position : current_position+window_len] #get the current window
		current_position = current_position + step_len #update current position

		X = abs(fft(x))       # get fft magnitude
		X = X[0:nfft]         # normalize fft
		X = X / len(X)
		if current_frame == 1:
			specgram = X ** 2
		else:
			specgram = np.vstack((specgram, X)) # spectrogram

		current_feature_vector = np.zeros((num_of_features, 1))
		current_feature_vector[0] = short_term_ZCR(x)  #zero crossing rate
		current_feature_vector[1] = short_term_Energy(x)
		current_feature_vector[2] = short_term_Energy_Entropy(x)
		[current_feature_vector[3], current_feature_vector[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
		current_feature_vector[5] = stSpectralEntropy(X) # spectral entropy
		current_feature_vector[6] = stSpectralRollOff(X, 0.90, fs)  # spectral rolloff

		short_term_features.append(current_feature_vector)

	short_term_features = np.concatenate(short_term_features, 1)
	# Concatenation refers to joining. This function is used to join two or more arrays of the same shape
	# along a specified axis. here the specified axis is 1.

	# if (PLOT):
	# 	fig, ax = plt.subplots()
    #     imgplot = plt.imshow(specgram.transpose()[::-1, :])
    #     Fstep = int(nfft / 5.0)
    #     FreqTicks = range(0, int(nfft) + Fstep, Fstep)
    #     FreqTicksLabels = [str(fs / 2 - int((f * fs) / (2 * nfft))) for f in FreqTicks]
    #     ax.set_yticks(FreqTicks)
    #     ax.set_yticklabels(FreqTicksLabels)
    #     TStep = current_frame/3
    #     TimeTicks = range(0, current_frame, TStep)
    #     TimeTicksLabels = ['%.2f' % (float(t * step_len) / fs) for t in TimeTicks]
    #     ax.set_xticks(TimeTicks)
    #     ax.set_xticklabels(TimeTicksLabels)
    #     ax.set_xlabel('time (secs)')
    #     ax.set_ylabel('freq (Hz)')
    #     imgplot.set_cmap('jet')
    #     plt.colorbar()



	FreqAxis = [((f + 1) * fs) / (2 * nfft) for f in range(specgram.shape[1])]
	TimeAxis = [(t * step_len) / fs for t in range(specgram.shape[0])]
	return short_term_features, specgram, FreqAxis, TimeAxis


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
  """Computes entropy of energy
  By further subdividing each frame into a set of sub-frames, we can calculate their
  respective short-term energies and treat them as probabilities, thus permitting
  us to calculate their entropy:
  """
  '''
  Entropy is the measure of randomness or disorderness of a system. So noise is considered
  to have a hight entropy as it is so random and dissordered. Music on the other hand is
  ordered and tends have very lower entropies. And entropy of music is lower that that of speech.
  '''

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



############################### SOME SPECTRAL FEATURES #####################################

def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = np.sum(X + eps)
    sumPrevX = np.sum(Xprev + eps)
    F = np.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F

def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid and spread of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)

def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = np.sum(X ** 2)            # total spectral energy

    subWinLength = int(np.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -np.sum(s*np.log2(s + eps))                                    # compute spectral entropy

    return En

def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)
