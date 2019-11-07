import librosa
import librosa.display
import numpy as np
import os


def save_spectrogram(filepath, start, stop, y, sr):
    """
    Input: .wav filepath, with start and stop time in seconds,
            y audio time series, sr samples rate of audio time
            series
    Output: Save a numpy file of mel spectrogram array of 
            dimension (n_mels, t)
    """
    s = librosa.feature.melspectrogram(y=y[sr * start:(sr * stop)],
                                       sr=sr, n_mels=64, fmax=sr / 2)
    s = librosa.power_to_db(s, ref=np.max)
    path_save = os.path.dirname(os.path.dirname(filepath)) + '/spectro/'
    if not (os.path.exists(path_save)):
        # create the directory you want to save to
        os.mkdir(path_save)
    file = filepath.split("/")[-1].rsplit("_",maxsplit=1)[0]
    print("Saving spectrogram for file: " + file + " window size between: " + str(i) + " sec to " + str(
        i + window_size) + " sec")

    np.save(path_save + file + '_' + str(start) + 'sto' + str(stop) + 's', s)


"""
Processing the audio files with a frame window size of 6 sec, advance of 3 sec, and generating the mel spectrogram
"""

audio_paths = ['/cache/rmishra/cc16_366a_converted/audio/']

window_size = 6
slide = 3
# file_count = 0
for path in audio_paths:
    for file in os.listdir(path):
        # file_count += 1
        filepath = path + file
        y, sr = librosa.load(filepath)
        length = int(len(y) / sr)
        remainder = length % window_size
        for i in range(0, length - remainder - window_size, slide):
            save_spectrogram(filepath, i, i + window_size, y, sr)
            # j = i + slide
            # if j + window_size < length - remainder:
            #     save_spectrogram(filepath, j, j + window_size, y, sr)
