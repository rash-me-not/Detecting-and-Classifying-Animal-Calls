import datetime
import os
import numpy as np
import librosa
import shutil
import pickle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
from AccelerometerFile import AccelerometerFile
from AudioFile import AudioFile
from network.network_train import create_model, create_model_using_z_axis, save_model, plot_accuracy, plot_loss, \
    plot_ROC, plot_class_ROC, \
    save_arch, save_folder
from preprocessing.create_data_groups import fetch_files_with_numcalls
from preprocessing.create_spectrograms import get_spectrogram
from preprocessing.create_labels import create_label_dataframe, create_label_matrix, find_label
from preprocessing.create_datasets import get_train_val_test, save_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


class HyenaCallDetection:

    def __init__(self, base_dir, spec_window_size, spec_advance):
        self.base_dir = base_dir
        self.window_size = spec_window_size
        self.slide = spec_advance
        self.dataset_dir = os.path.join(base_dir, 'dataset')
        self.combined_dir = os.path.join(base_dir, 'combined')

    def main(self):
        hyena_data = ['cc16_352a', 'cc16_352b', 'cc16_354a', 'cc16_360a', 'cc16_366a']

        for hyena_recording in hyena_data:

            # Fetch all the audio data with greater than 1 hyena call
            audio_path = os.path.join(self.base_dir, 'cc16_ML',
                                      hyena_recording)  # audio_path:'/cache/rmishra/cc16_ML/cc16_352a'
            # file_list = fetch_files_with_numcalls(audio_path, 1)

            hyena_rec_converted = os.path.join(self.base_dir, hyena_recording + '_converted')

            # for file_aud, file_acc in zip(file_list['audio'], file_list['acc']):
            #     self.save_spec(os.path.join(audio_path, file_aud), hyena_rec_converted, audio_path)
            #     self.save_spec(os.path.join(audio_path, file_acc), hyena_rec_converted, audio_path)


        # self.save_dataset();
        print("Saved dataset")

        x_train_aud = np.load(os.path.join(self.dataset_dir,"dataset_aud/x_train.npy"));
        x_train_acc_ch0 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_0/x_train.npy"))
        x_train_acc_ch1 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_1/x_train.npy"))
        x_train_acc_ch2 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_2/x_train.npy"))

        y_train_aud = np.load(os.path.join(self.dataset_dir,"dataset_aud/y_train_aud.npy"))
        y_train_foc = np.load(os.path.join(self.dataset_dir, "dataset_aud/y_train_foc.npy"))

        x_val_aud = np.load(os.path.join(self.dataset_dir,"dataset_aud/x_val.npy"))
        x_val_acc_ch0 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_0/x_val.npy"))
        x_val_acc_ch1 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_1/x_val.npy"))
        x_val_acc_ch2 = np.load(os.path.join(self.dataset_dir,"dataset_acc_ch_2/x_val.npy"))

        y_val_aud = np.load(os.path.join(self.dataset_dir,"dataset_aud/y_val_aud.npy"))
        y_val_foc = np.load(os.path.join(self.dataset_dir,"dataset_aud/y_val_foc.npy"))

        print("Creating model")
        # Train the RCNN model
        # model = create_model(x_train_aud, x_train_acc_ch0, x_train_acc_ch1, x_train_acc_ch2,
        #                      filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)

        model = create_model_using_z_axis(x_train_aud, x_train_acc_ch2,
                             filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)

        print(model.summary())
        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
        epochs = 1
        batch_size = 32
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
        reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                           mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

        print("TRaining model")
        # model_fit = model.fit([x_train_aud, x_train_acc_ch0, x_train_acc_ch1, x_train_acc_ch2], [y_train_aud, y_train_foc],
        #                       epochs=epochs, batch_size=batch_size,
        #                       validation_data=([x_val_aud, x_val_acc_ch0, x_val_acc_ch1, x_val_acc_ch2], [y_val_aud, y_val_foc]),
        #                       shuffle=True,
        #                       callbacks=[early_stopping, reduce_lr_plat])

        model_fit = model.fit([x_train_aud, x_train_acc_ch2], [y_train_aud, y_train_foc],
                              epochs=epochs, batch_size=batch_size,
                              validation_data=([x_val_aud, x_val_acc_ch2], [y_val_aud, y_val_foc]),
                              shuffle=True,
                              callbacks=[early_stopping, reduce_lr_plat])

        # model_fit = load_model('saved_models/model_2019-11-17_03:19:50.898753_network_train/savedmodel.h5')
        #
        # # with open('saved_models/model_2019-11-17_03:19:50.898753_network_train/history.pickle', 'rb') as handle:  # loading old history
        # #     history = pickle.load(handle)
        print("MOdel TRained")
        date_time = datetime.datetime.now()
        sf = save_folder(date_time)
        self.create_save_folder(sf)

        print("Saving Model")
        model.save(sf + '/savedmodel' + '.h5')
        with open(sf + '/history.pickle', 'wb') as f:
            pickle.dump(model_fit.history, f)

        plot_accuracy(model_fit, sf)
        plot_loss(model_fit, sf)
        # plot_ROC(model, [x_val_aud, x_val_acc_ch0, x_val_acc_ch1, x_val_acc_ch2], y_val_aud, sf)
        # plot_class_ROC(model, [x_val_aud, x_val_acc_ch0, x_val_acc_ch1, x_val_acc_ch2], y_val_aud, sf)
        save_arch(model, sf)

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
        SAVE_FILE = '1plus_dataset.npy'
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
            # save_data(dataset, SAVE_FILE, dataset_folder)
            train_ratio = 0.75
            val_ratio = 0.15
            train_test_data = get_train_val_test(dataset, train_ratio, val_ratio)
            x_train, y_train_aud, y_train_foc, train_files = train_test_data.get("train")
            x_val, y_val_aud, y_val_foc, val_files = train_test_data.get("val")
            x_test, y_test_aud, y_test_foc, test_files = train_test_data.get("test")

            dataset_fname_dict = {'x_train': x_train, 'y_train_aud': y_train_aud, 'y_train_foc': y_train_foc, 'train_files': train_files,
                                  'x_val': x_val, 'y_val_aud': y_val_aud, 'y_val_foc': y_val_foc, 'val_files': val_files,
                                  'x_test': x_test, 'y_test_aud': y_test_aud, 'y_test_foc': y_test_foc, 'test_files': test_files }

            for fname, file in dataset_fname_dict.items():
                if all(elem is not None for elem in file):
                    save_data(file, fname, dataset_folder)

    def create_save_folder(self, save_folder):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

if __name__ == "__main__":
    base_dir = "D:\Rashmita"
    spec_window_size = 6
    slide = 3

    detection = HyenaCallDetection(base_dir, spec_window_size, slide)
    detection.main()
