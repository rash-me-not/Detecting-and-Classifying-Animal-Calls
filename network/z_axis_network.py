import datetime
import pickle

from network.network_train import NetworkTrain
import numpy as np
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D

class ZAxisNetwork(NetworkTrain):

    def __init__(self, dataset_dir, epochs, batch_size):
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size

    def train_network(self):
        x_train_aud = np.load(os.path.join(self.dataset_dir, "dataset_aud/x_train.npy"));
        x_train_acc_ch2 = np.load(os.path.join(self.dataset_dir, "dataset_acc_ch_2/x_train.npy"))

        y_train_aud = np.load(os.path.join(self.dataset_dir, "dataset_aud/y_train_aud.npy"))
        y_train_foc = np.load(os.path.join(self.dataset_dir, "dataset_aud/y_train_foc.npy"))

        x_val_aud = np.load(os.path.join(self.dataset_dir, "dataset_aud/x_val.npy"))
        x_val_acc_ch2 = np.load(os.path.join(self.dataset_dir, "dataset_acc_ch_2/x_val.npy"))

        y_val_aud = np.load(os.path.join(self.dataset_dir, "dataset_aud/y_val_aud.npy"))
        y_val_foc = np.load(os.path.join(self.dataset_dir, "dataset_aud/y_val_foc.npy"))

        print("Creating model")
        # Train the RCNN model
        model = self.create_model([x_train_aud, x_train_acc_ch2])

        print(model.summary())
        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
        reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                           mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

        print("TRaining model")

        model_fit = model.fit([x_train_aud, x_train_acc_ch2], [y_train_aud, y_train_foc],
                              epochs=self.epochs, batch_size=self.batch_size,
                              validation_data=([x_val_aud, x_val_acc_ch2], [y_val_aud, y_val_foc]),
                              shuffle=True,
                              callbacks=[early_stopping, reduce_lr_plat])

        # model_fit = load_model('saved_models/model_2019-12-22_18:13:21.153329_network_train/savedmodel.h5')
        #
        # with open('saved_models/model_2019-12-22_18:13:21.153329_network_train/history.pickle', 'rb') as handle:  # loading old history
        #     history = pickle.load(handle)

        print("MOdel TRained")
        date_time = datetime.datetime.now()
        sf = self.save_folder_w_datetime(date_time).replace(":", "_")
        self.create_save_folder(sf)

        print("Saving Model")
        model.save(sf + '/savedmodel' + '.h5')
        with open(sf + '/history.pickle', 'wb') as f:
            pickle.dump(model_fit.history, f)

        self.plot_accuracy(model_fit, sf)
        self.plot_loss(model_fit, sf)
        self.plot_ROC(model, [x_val_aud, x_val_acc_ch2], [y_val_aud, y_val_foc], sf)
        self.plot_class_ROC(model, [x_val_aud, x_val_acc_ch2], [y_val_aud, y_val_foc], sf)
        self.save_arch(model, sf)


    def create_model(self, xtrain, filters=128, gru_units=128, dense_neurons=1024, dropout=0.5):
        """
        Outputs a non sequntial keras model
        filters = number of filters in each convolutional layer
        gru_units = number of gru units in each recurrent layer
        dense_neurons = number of neurons in the time distributed dense layers
        dropout = dropout rate used throughout the model
        """
        x_train_aud = xtrain[0]
        x_train_acc_ch2 = xtrain[1]

        inp_aud = Input(shape=(x_train_aud.shape[1], x_train_aud.shape[2], 1))
        inp_acc_2 = Input(shape=(x_train_acc_ch2.shape[1], x_train_acc_ch2.shape[2], 1))

        aud = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp_aud)
        aud = MaxPooling2D(pool_size=(1, 5))(aud)
        aud = Conv2D(filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Conv2D(filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Model(inputs=inp_aud, outputs=aud)

        acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp_acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 5))(acc_2)
        acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        acc_2 = Model(inputs=inp_acc_2, outputs=acc_2)

        combined = concatenate([aud.output, acc_2.output])
        combined = Reshape((x_train_aud.shape[-3], -1))(combined)
        dense_foctype_1 = TimeDistributed(Dense(dense_neurons, activation='relu'))(combined)
        drop_foctype_1 = Dropout(rate=dropout)(dense_foctype_1)
        dense_foctype_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_foctype_1)
        drop_foctype_2 = Dropout(rate=dropout)(dense_foctype_2)
        output_foctype = TimeDistributed(Dense(3, activation='softmax'), name="output_foctype")(drop_foctype_2)

        rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout,
                                  recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(combined)
        rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout,
                                  recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)

        dense_1 = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=dropout)(dense_3)
        output_aud = TimeDistributed(Dense(9, activation='sigmoid'), name="output_aud")(drop_3)
        model = Model(inputs=[aud.input, acc_2.input], outputs=[output_aud, output_foctype])
        return model

