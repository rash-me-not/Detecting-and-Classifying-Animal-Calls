import datetime
import os
import numpy as np
import librosa
import shutil
import pickle
import tensorflow as tf
from AccelerometerFile import AccelerometerFile
from AudioFile import AudioFile
from network.z_axis_network import ZAxisNetwork
from network.three_axis_network import ThreeAxisNetwork
from preprocessing.create_data_groups import fetch_files_with_numcalls
from preprocessing.create_spectrograms import get_spectrogram
from preprocessing.create_labels import create_label_dataframe, create_label_matrix, find_label
from preprocessing.create_datasets import get_train_val_test, save_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


class HyenaCallDetection:

    def __init__(self, base_dir, spec_window_size, spec_advance, only_z_axis):
        self.base_dir = base_dir
        self.window_size = spec_window_size
        self.slide = spec_advance
        self.dataset_dir = os.path.join(base_dir, 'dataset')
        self.combined_dir = os.path.join(base_dir, 'combined')
        self.only_z_axis = only_z_axis

    def main(self):
        # hyena_data = ['cc16_352a', 'cc16_352b', 'cc16_354a', 'cc16_360a', 'cc16_366a']
        hyena_data = ['cc16_352a_test']

        for hyena_recording in hyena_data:
            # Fetch all the audio data with greater than 1 hyena call
            audio_path = os.path.join(self.base_dir, 'cc16_ML',
                                      hyena_recording)  # audio_path:'/cache/rmishra/cc16_ML/cc16_352a'
            file_list = fetch_files_with_numcalls(audio_path, 1)

            hyena_rec_converted = os.path.join(self.base_dir, hyena_recording + '_converted')

            for file_aud, file_acc in zip(file_list['audio'], file_list['acc']):
                self.save_spec(os.path.join(audio_path, file_aud), hyena_rec_converted, audio_path)
                self.save_spec(os.path.join(audio_path, file_acc), hyena_rec_converted, audio_path)

        self.save_dataset();
        print("Saved dataset")

        epochs = 1
        batch_size = 64

        # Train the network based on the boolean flag that indicates the model for 3 axis input or z-axis input
        training_axis = ZAxisNetwork(self.dataset_dir, epochs, batch_size) if self.only_z_axis else ThreeAxisNetwork(
            self.dataset_dir, epochs, batch_size)
        training_axis.train_network()

    def save_spec(self, filepath, converted_dir, audio_path):
        y, sr = librosa.load(filepath, sr=None, mono=False)

        # Reshaping the Audio file (mono) to deal with all wav files similarly
        if y.ndim == 1:
            y = y.reshape(1, -1)

        for ch in range(y.shape[0]):
            length = int(len(y[ch]) / sr)
            remainder = length % self.window_size

            audio = AudioFile(self.base_dir, converted_dir)
            acc = AccelerometerFile(self.base_dir, converted_dir, ch)

            file_type = audio if y.shape[0] == 1 else acc

            for i in range(0, length - remainder - self.window_size, self.slide):
                begin_time = i
                end_time = i + self.window_size

                s_db = get_spectrogram(begin_time, end_time, y[ch], sr)
                # Extracting file identifier from the filepath
                # Example: (i.e.'cc16_352a_14401s' from path "'/cache/rmishra/cc16_ML/cc16_352a/cc16_352a_14401s_acc.wav'")
                # for saving spec and label files with begin and end timestamp
                file = filepath.split("/")[-1].rsplit("_", maxsplit=1)[0]

                # fetch the label txt file against the file identifier and create a label dataframe for calls between
                # the start and end timestamp
                call = fetch_files_with_numcalls(audio_path, 1).loc[file]['calls']
                timesteps = s_db.shape[1]
                timesteps_per_second = timesteps / self.window_size
                df = create_label_dataframe(os.path.join(audio_path, call),
                                            begin_time,
                                            end_time,
                                            self.window_size,
                                            timesteps_per_second)

                # one hot encoding the label information for the audio data in a spec frame
                label_matrix = create_label_matrix(df, timesteps)

                if 1 in label_matrix[0][:8, :]:
                    print("Saving spectrogram: " + filepath + " " + str(begin_time) + " to " + str(end_time))
                    file_type.save_spec_label(s_db, begin_time, end_time, file, label_matrix)

    def save_dataset(self):
        # Cumulating the combined spectrogram and label information in 1plus_dataset.npy
        for subfolder in os.listdir(self.combined_dir):
            dataset_folder = os.path.join(self.dataset_dir, 'dataset' + subfolder.split('label')[1])
            self.create_save_folder(dataset_folder)
            dataset = []
            files = os.listdir(os.path.join(self.combined_dir, subfolder))
            for i in range(len(files)):
                current_file = np.load(os.path.join(self.combined_dir, subfolder, files[i]))
                dataset.append([files[i], current_file])
            dataset = np.array(dataset)

            print("Created Dataset")
            train_ratio = 0.75
            val_ratio = 0.15
            train_test_data = get_train_val_test(dataset, train_ratio, val_ratio)
            x_train, y_train_aud, y_train_foc, train_files = train_test_data.get("train")
            x_val, y_val_aud, y_val_foc, val_files = train_test_data.get("val")
            x_test, y_test_aud, y_test_foc, test_files = train_test_data.get("test")

            dataset_fname_dict = {'x_train': x_train, 'y_train_aud': y_train_aud, 'y_train_foc': y_train_foc,
                                  'train_files': train_files,
                                  'x_val': x_val, 'y_val_aud': y_val_aud, 'y_val_foc': y_val_foc,
                                  'val_files': val_files,
                                  'x_test': x_test, 'y_test_aud': y_test_aud, 'y_test_foc': y_test_foc,
                                  'test_files': test_files}

            for fname, file in dataset_fname_dict.items():
                if all(elem is not None for elem in file):
                    save_data(file, fname, dataset_folder)

    def create_save_folder(self, save_folder):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)


if __name__ == "__main__":
    base_dir = "/cache/rmishra"
    spec_window_size = 6
    slide = 3
    only_z_axis = True

    detection = HyenaCallDetection(base_dir, spec_window_size, slide, only_z_axis)
    detection.main()
