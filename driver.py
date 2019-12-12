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
from network.network_train import create_model, save_model, plot_accuracy, plot_loss, plot_ROC, plot_class_ROC, \
    save_arch, save_folder, create_save_folder
from preprocessing.create_data_groups import fetch_files_with_numcalls
from preprocessing.create_spectrograms import get_spectrogram
from preprocessing.create_labels import create_label_dataframe, create_label_matrix, find_label
from preprocessing.create_datasets import get_train_val_test, save_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

def save_spec(window_size, slide,  filepath, spec_dir, combined_dir, audio_path):
        y, sr = librosa.load(filepath, sr=None, mono=False)
        if y.ndim==1:
            y = y.reshape(1, -1)

        for ch in range(y.shape[0]):
            length = int(len(y[ch]) / sr)
            remainder = length % window_size
            spec_path = os.path.join(spec_dir, "spec_aud") if y.shape[0]==1 else os.path.join(spec_dir,
                                                                                            "spec_acc_ch_" + str(ch))
            combined_spec_label = os.path.join(combined_dir, "combined_spec_label_aud") if y.shape[0]==1 else os.path.join(combined_dir, "combined_spec_label_acc_ch_"+str(ch))

            create_save_folder(spec_path)
            create_save_folder(combined_spec_label)

            for i in range(0, length - remainder - window_size, slide):
                print("Saving spectrogram: " + filepath + " " + str(i) + " to " + str(i + window_size))
                begin_time = i
                end_time = i+window_size

                s_db = get_spectrogram(begin_time, end_time, y[ch], sr)

                file = filepath.split("/")[-1].rsplit("_", maxsplit=1)[0]
                spec_file =  file + '_' + str(begin_time) + 'sto' + str(end_time) + 's'
                np.save(os.path.join(spec_path,spec_file), s_db)

                label = fetch_files_with_numcalls(audio_path,1).loc[file]['labels']
                timesteps = s_db.shape[1]
                timesteps_per_second = timesteps / window_size
                df = create_label_dataframe(os.path.join(audio_path, label),
                                            begin_time,
                                            end_time,
                                            window_size,
                                            timesteps_per_second)
                label_matrix = create_label_matrix(df,
                                                   timesteps)  # one hot encoding the label information for the audio data in a spec frame

                if 1 in label_matrix[0:8, :]:  # Saving the one hot encoded Label file if there is at least one call present in the 6 sec segment
                    label_path = os.path.join(spec_path, spec_file + 'LABEL')
                    print("Saving file: {}".format(label_path))
                    np.save(label_path, label_matrix)
                    np.savetxt(label_path, label_matrix, delimiter=",")
                    combined = np.array((s_db, label_matrix))
                    np.save(os.path.join(combined_spec_label, spec_file + 'SPEC_LAB'),
                            combined)  # Combine and save the spec and label information

def main():
    base_dir = "/cache/rmishra"
    combined_spec_label = os.path.join(base_dir, "combined")
    hyena_data = ['cc16_352a', 'cc16_352b', 'cc16_366a', 'cc16_354a', 'cc16_360a']

    for hyena_recording in hyena_data:

        # Fetch all the audio data with greater than 1 hyena call
        audio_path = os.path.join(base_dir, 'cc16_ML', hyena_recording)  # audio_path:'/cache/rmishra/cc16_ML/cc16_352a'
        file_list = fetch_files_with_numcalls(audio_path, 1)

        hyena_rec_converted = os.path.join(base_dir, hyena_recording + '_converted')


        # Processing the audio files with a frame window size of 6 sec, advance of 3 sec, and generating the mel spectrogram
        window_size = 6
        slide = 3

        for file_aud, file_acc in zip(file_list['audio'], file_list['acc']):
            save_spec(window_size, slide, os.path.join(audio_path, file_acc), hyena_rec_converted, combined_spec_label, audio_path)
            save_spec(window_size, slide, os.path.join(audio_path,file_aud), hyena_rec_converted, combined_spec_label, audio_path)



    # Cumulating the combined spectrogram and label information in 1plus_dataset.npy
    # SAVE_FILE = '1plus_dataset.npy'
    # dataset_path = "/cache/rmishra/dataset"
    # for subfolder in os.listdir(combined_spec_label):
    #     dataset_folder = os.path.join(dataset_path, 'dataset' + subfolder.split('label')[1])
    #     dataset = []
    #     files = os.listdir(os.path.join(combined_spec_label, subfolder))
    #     for i in range(len(files)):
    #         current_file = np.load(os.path.join(combined_spec_label, subfolder, files[i]))
    #         dataset.append([files[i], current_file])
    #     dataset = np.array(dataset)
    #
    #     print("Created Dataset")
    #     # save_data(dataset, SAVE_FILE, dataset_folder)
    #     train_ratio = 0.75
    #     val_ratio = 0.15
    #     train_test_data = get_train_val_test(dataset, train_ratio, val_ratio)
    #     x_train, y_train, train_files = train_test_data.get("train")
    #     x_val, y_val, val_files = train_test_data.get("val")
    #     x_test, y_test, test_files = train_test_data.get("test")
    #     save_data(x_train, 'x_train.npy', dataset_folder)
    #     save_data(y_train, 'y_train.npy', dataset_folder)
    #     save_data(train_files, 'train_files.npy', dataset_folder)
    #     save_data(x_val, 'x_val.npy', dataset_folder)
    #     save_data(y_val, 'y_val.npy', dataset_folder)
    #     save_data(val_files, 'val_files.npy', dataset_folder)
    #     save_data(x_test, 'x_test.npy', dataset_folder)
    #     save_data(y_test, 'y_test.npy', dataset_folder)
    #     save_data(test_files, 'test_files.npy', dataset_folder)
    #
    # print("Saved dataset")
    x_train_aud = np.load("/cache/rmishra/dataset/dataset_aud/x_train.npy")
    x_train_acc_ch0 = np.load("/cache/rmishra/dataset/dataset_acc_ch_0/x_train.npy")
    x_train_acc_ch1 = np.load("/cache/rmishra/dataset/dataset_acc_ch_1/x_train.npy")
    x_train_acc_ch2 = np.load("/cache/rmishra/dataset/dataset_acc_ch_2/x_train.npy")

    y_train_aud = np.load("/cache/rmishra/dataset/dataset_aud/y_train.npy")

    x_val_aud = np.load("/cache/rmishra/dataset/dataset_aud/x_val.npy")
    x_val_acc_ch0 = np.load("/cache/rmishra/dataset/dataset_acc_ch_0/x_val.npy")
    x_val_acc_ch1 = np.load("/cache/rmishra/dataset/dataset_acc_ch_1/x_val.npy")
    x_val_acc_ch2 = np.load("/cache/rmishra/dataset/dataset_acc_ch_2/x_val.npy")

    y_val_aud = np.load("/cache/rmishra/dataset/dataset_aud/y_val.npy")

    print("Creating model")
    # Train the RCNN model
    model = create_model(x_train_aud, x_train_acc_ch0, x_train_acc_ch1, x_train_acc_ch2,
                         filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)

    print(model.summary())
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    epochs = 20
    batch_size = 32
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
    reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                       mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

    print("TRaining model")
    model_fit = model.fit([x_train_aud,x_train_acc_ch0, x_train_acc_ch1, x_train_acc_ch2],  y_train_aud, epochs=epochs, batch_size=batch_size,
                          validation_data=([x_val_aud, x_val_acc_ch0, x_val_acc_ch1, x_val_acc_ch2], y_val_aud), shuffle=True,
                          callbacks=[early_stopping, reduce_lr_plat])

    # model_fit = load_model('saved_models/model_2019-11-17_03:19:50.898753_network_train/savedmodel.h5')
    #
    # with open('saved_models/model_2019-11-17_03:19:50.898753_network_train/history.pickle', 'rb') as handle:  # loading old history
    #     history = pickle.load(handle)
    print("MOdel TRained")
    date_time = datetime.datetime.now()
    sf = save_folder(date_time)
    create_save_folder(sf)

    print("Saving Model")
    model.save(sf + '/savedmodel' + '.h5')
    with open(sf + '/history.pickle', 'wb') as f:
        pickle.dump(model_fit.history, f)

    plot_accuracy(model_fit, sf)
    plot_loss(model_fit, sf)
    # plot_ROC(model, x_val, y_val, sf)
    # plot_class_ROC(model, x_val, y_val, sf)
    # save_arch(model, sf)


if __name__ == "__main__":
    main()
