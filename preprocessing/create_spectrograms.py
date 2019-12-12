import librosa
import librosa.display
import numpy as np
import os


def get_spectrogram(start, stop, y, sr):
    """
    Input: .wav filepath, with start and stop time in seconds,
            y audio time series, sr samples rate of audio time
            series
    Output: Save a numpy file of mel spectrogram array of 
            dimension (n_mels, t)
    """
    nfft = int(0.020 * sr)
    hop = int(0.010 * sr)

    s = librosa.feature.melspectrogram(y=np.asfortranarray(y[sr * start:(sr * stop)]),
                                       sr=sr, n_mels=64, fmax=sr / 2, hop_length=hop, window='hann', win_length=nfft)
    s_db = librosa.power_to_db(s, ref=np.max)

    return s_db

