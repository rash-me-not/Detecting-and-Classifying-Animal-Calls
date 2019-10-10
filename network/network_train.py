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
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input
from keras.models import Model 
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D 
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_curve, auc


def create_model(filters, gru_units, dense_neurons, dropout):
    """
    Outputs a non sequntial keras model
    filters = number of filters in each convolutional layer
    gru_units = number of gru units in each recurrent layer
    dense_neurons = number of neurons in the time distributed dense layers
    dropout = dropout rate used throughout the model
    """
    inp = Input(shape=(259, 64, 1))
    c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
    mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
    c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
    mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
    c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
    mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

    reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

    rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
    rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)
    
    dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
    drop_1 = Dropout(rate=dropout)(dense_1)
    dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
    drop_2 = Dropout(rate=dropout)(dense_2)
    dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
    drop_3 = Dropout(rate=dropout)(dense_3)
    output = TimeDistributed(Dense(8, activation='sigmoid'))(drop_3)

    model = Model(inp, output)
    return model


def save_folder(date_time):
    date_now = str(date_time.date())
    time_now = str(date_time.time())
    sf = "saved_models/model_" + date_now + "_" + time_now + "_" + os.path.basename(__file__).split('.')[0]
    return sf


def create_save_folder(save_folder):  
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

        
def save_model(save_folder):
    """
    Output: Saves dictionary of model training history as a pickle file.
    """
    model.save(save_folder + '/savedmodel' + '.h5')
    with open(save_folder + '/history.pickle', 'wb') as f:
        pickle.dump(model_fit.history, f)


def plot_accuracy(model_fit, save_folder):
    """
    Output: Plots and saves graph of accuracy at each epoch. 
    """
    train_acc = model_fit.history['binary_accuracy']
    val_acc = model_fit.history['val_binary_accuracy']
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
    

def plot_loss(model_fit, save_folder):
    """
    Output: Plots and saves graph of loss at each epoch. 
    """
    train_loss = model_fit.history['loss']
    val_loss = model_fit.history['val_loss']
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
    class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
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
    with open(save_folder + '/architecture.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))


os.environ["CUDA_VISIBLE_DEVICES"]="2" # select GPU
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.75 # set use %
tf.Session(config=config)

# load train and validation datasets
x_train = np.load('datasets/x_train.npy')
x_val = np.load('datasets/x_val.npy')
y_train = np.load('datasets/y_train.npy')
y_val = np.load('datasets/y_val.npy')

model = create_model(filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)
print(model.summary())
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
epochs = 2500
batch_size = 256
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

model_fit = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_val, y_val), shuffle=True,
                      callbacks=[early_stopping, reduce_lr_plat])
date_time = datetime.datetime.now()
sf = save_folder(date_time)
create_save_folder(sf)
save_model(sf)
plot_accuracy(model_fit, sf)
plot_loss(model_fit, sf)
plot_ROC(model, x_val, y_val, sf)
plot_class_ROC(model, x_val, y_val, sf)
save_arch(model, sf)
