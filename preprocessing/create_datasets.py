import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split


SAVE_PATH = '/datasets/'

# make x_train, x_val, y_train, y_val
dataset = np.load('/datasets/1plus_dataset.npy')
x = dataset[:, 0]
y = dataset[:, 1]
# reformat x to (n, timesteps, mel bands, 1)
x = np.expand_dims(np.moveaxis(np.stack(x), 1, -1), axis=3)
# reformat y to (n, timesteps, classes)
y = np.moveaxis(np.stack(y), 1, -1)
# split data into test, train and validation
TRAIN = 0.75
VALIDATION = 0.15
TEST = 0.10

# create train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - TRAIN)

# create val and test set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=TEST/(TEST + VALIDATION)) 

# save files
np.save(SAVE_PATH + 'x_train.npy', x_train)
np.save(SAVE_PATH + 'x_val.npy', x_val)
np.save(SAVE_PATH + 'x_test.npy', x_test)
np.save(SAVE_PATH + 'y_train.npy', y_train)
np.save(SAVE_PATH + 'y_val.npy', y_val)
np.save(SAVE_PATH + 'y_test.npy', y_test)