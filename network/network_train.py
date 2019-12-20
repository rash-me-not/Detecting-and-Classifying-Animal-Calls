import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pickle
from keras.models import load_model
from keras.layers import Input, Flatten
from keras.models import Model 
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from sklearn.metrics import roc_curve, auc
from datetime import datetime

def create_model(x_train_aud, x_train_acc_ch0, x_train_acc_ch1, x_train_acc_ch2, filters, gru_units, dense_neurons, dropout):
    """
    Outputs a non sequntial keras model
    filters = number of filters in each convolutional layer
    gru_units = number of gru units in each recurrent layer
    dense_neurons = number of neurons in the time distributed dense layers
    dropout = dropout rate used throughout the model
    """
    inp_aud = Input(shape=(x_train_aud.shape[1], x_train_aud.shape[2], 1))
    inp_acc_0 = Input(shape=(x_train_acc_ch0.shape[1], x_train_acc_ch0.shape[2], 1))
    inp_acc_1 = Input(shape=(x_train_acc_ch1.shape[1], x_train_acc_ch1.shape[2], 1))
    inp_acc_2 = Input(shape=(x_train_acc_ch2.shape[1], x_train_acc_ch2.shape[2], 1))

    aud = Conv2D(filters, (3,3), padding='same', activation='relu')(inp_aud)
    aud = MaxPooling2D(pool_size=(1,5))(aud)
    aud = Conv2D(filters, (3,3), padding='same', activation='relu')(aud)
    aud = MaxPooling2D(pool_size=(1,2))(aud)
    aud = Conv2D(filters, (3,3), padding='same', activation='relu')(aud)
    aud = MaxPooling2D(pool_size=(1,2))(aud)
    aud = Reshape((x_train_aud.shape[-3], -1))(aud)
    aud = Model(inputs=inp_aud, outputs=aud)


    acc_0 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp_acc_0)
    acc_0 = MaxPooling2D(pool_size=(1,5))(acc_0)
    acc_0 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_0)
    acc_0 = MaxPooling2D(pool_size=(1,2))(acc_0)
    acc_0 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_0)
    acc_0 = MaxPooling2D(pool_size=(1,2))(acc_0)
    acc_0 = Reshape((x_train_acc_ch0.shape[-3], -1))(acc_0)
    acc_0 = Model(inputs=inp_acc_0, outputs=acc_0)

    acc_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp_acc_1)
    acc_1 = MaxPooling2D(pool_size=(1,5))(acc_1)
    acc_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_1)
    acc_1 = MaxPooling2D(pool_size=(1,2))(acc_1)
    acc_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_1)
    acc_1 = MaxPooling2D(pool_size=(1,2))(acc_1)
    acc_1 = Reshape((x_train_acc_ch0.shape[-3], -1))(acc_1)
    acc_1 = Model(inputs=inp_acc_1, outputs=acc_1)

    acc_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp_acc_2)
    acc_2 = MaxPooling2D(pool_size=(1,5))(acc_2)
    acc_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_2)
    acc_2 = MaxPooling2D(pool_size=(1,2))(acc_2)
    acc_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(acc_2)
    acc_2 = MaxPooling2D(pool_size=(1,2))(acc_2)
    acc_2 = Reshape((x_train_acc_ch0.shape[-3], -1))(acc_2)
    acc_2 = Model(inputs=inp_acc_2, outputs=acc_2)

    combined = concatenate([aud.output, acc_0.output, acc_1.output, acc_2.output])

    # flattened = Flatten()(combined)
    # dense_foctype_1 = Dense(dense_neurons, activation='relu')(flattened)
    # drop_foctype_1 = Dropout(rate=dropout)(dense_foctype_1)
    # dense_foctype_2 = Dense(dense_neurons, activation='relu')(drop_foctype_1)
    # drop_foctype_2 = Dropout(rate=dropout)(dense_foctype_2)
    # output_foctype = Dense(3, activation='softmax')(drop_foctype_2)

    rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(combined)
    rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)
    
    dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
    drop_1 = Dropout(rate=dropout)(dense_1)
    dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
    drop_2 = Dropout(rate=dropout)(dense_2)
    dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
    drop_3 = Dropout(rate=dropout)(dense_3)
    output_aud = TimeDistributed(Dense(9, activation='sigmoid'))(drop_3)
    output_foctype = Dense(3, activation='softmax')(drop_3)
    model = Model(inputs=[aud.input, acc_0.input, acc_1.input, acc_2.input], outputs=[output_aud,output_foctype])
    return model


def save_folder(date_time):
    date_now = str(date_time.date())
    time_now = str(date_time.time())
    sf = "saved_models/model_" + date_now + "_" + time_now + "_" + os.path.basename(__file__).split('.')[0]
    return sf

        
def save_model(save_folder, model, model_fit):
    """
    Output: Saves dictionary of model training history as a pickle file.
    """
    model.save(save_folder + '/savedmodel' + '.h5')
    with open(save_folder + '/history.pickle', 'wb') as f:
        pickle.dump(model_fit.history, f)


def plot_accuracy(model_fit, save_folder, history=None):
    """
    Output: Plots and saves graph of accuracy at each epoch. 
    """

    train_acc = history['binary_accuracy'] if history is not None else model_fit.history['binary_accuracy']
    val_acc = history['val_binary_accuracy'] if history is not None else model_fit.history['val_binary_accuracy']
    epoch_axis = np.arange(1, len(train_acc) + 1)
    plt.title('Train vs Validation Accuracy')
    plt.plot(epoch_axis, train_acc, 'b', label='Train Acc')
    plt.plot(epoch_axis, val_acc,'r', label='Val Acc')
    plt.xlim([1, len(train_acc)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_acc) / 10) + 0.5)))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/accuracy.png')
    plt.show()
    plt.close()
    

def plot_loss(model_fit, save_folder, history=None):
    """
    Output: Plots and saves graph of loss at each epoch. 
    """

    train_loss = history['loss'] if history is not None else model_fit.history['loss']
    val_loss = history['val_loss'] if history is not None else model_fit.history['val_loss']
    epoch_axis = np.arange(1, len(train_loss) + 1)
    plt.title('Train vs Validation Loss')
    plt.plot(epoch_axis, train_loss, 'b', label='Train Loss')
    plt.plot(epoch_axis, val_loss,'r', label='Val Loss')
    plt.xlim([1, len(train_loss)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_loss) / 10) + 0.5)))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/loss.png')
    plt.show()
    plt.close()


def plot_ROC(model, x_val, y_val, save_folder):
    """
    Output: Plots and saves overall ROC graph
    for the validation set.
    """

    predicted = model.predict(x_val).ravel()
    actual = y_val.ravel()
    fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic Overall')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_folder + '/ROC.png')
    plt.show()
    plt.close()


def plot_class_ROC(model, x_val, y_val, save_folder):
    """
    Output: Plots and saves ROC graphs
    for the validation set.
    """
    class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP','OTH']
    for i in range(len(class_names)):
        predicted = model.predict(x_val)[:,:,i].ravel()
        actual = y_val[:,:,i].ravel()
        fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic ' + class_names[i])
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(save_folder + '/class_ROC_' + class_names[i] + '.png')
        plt.show()
        plt.close()


def save_arch(model, save_folder):
    with open(save_folder + '/archiecture.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def count_labels(file):
    count = {}
    sound_label = {0: 'GIG', 1: 'SQL', 2: 'GRL', 3: 'GRN', 4: 'SQT', 5: 'MOO', 6: 'RUM', 7: 'WHP', 8:'OTH'}
    y = np.load(file)
    data = np.where(y == 1)[2]
    for label_idx in data:
        label = sound_label[label_idx]
        count[label] = count.get(label, 0) + 1
    return count


