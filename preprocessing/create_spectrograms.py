import librosa
import numpy as np
import sys
import os

def save_spectrogram(filepath, start, stop, y, sr):
    """
    Input: .wav filepath, with start and stop time in seconds,
            y audio time series, sr samples rate of audio time
            series
    Output: Save a numpy file of mel spectrogram array of 
            dimension (n_mels, t)
    """
    S = librosa.feature.melspectrogram(y=y[sr*start:(sr*stop)],
                                       sr=sr, n_mels=64, fmax=sr/2) 
    path_save = os.path.dirname(os.path.dirname(filepath)) + '/spectro/'
    file = filepath.split('/')[-1][:-7]
    np.save(path_save + file + '_' + str(start) + 'sto' + str(stop) + 's', S)


audio_paths = ['cc16_352a_converted/audio/',
              'cc16_352b_converted/audio/',
              'cc16_354a_converted/audio/',
              'cc16_360a_converted/audio/',
              'cc16_366a_converted/audio/']

window_size = 6
slide = 3

for path in audio_paths:
    for file in os.listdir(path):
        filepath = path + file
        y, sr = librosa.load(filepath)
        length = int(len(y) / sr)
        remainder = length % window_size
        for i in range(0, length - remainder - window_size, slide):
            save_spectrogram(filepath, i, i + window_size, y, sr)
            
