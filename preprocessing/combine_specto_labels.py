import os
import numpy as np
import sys

LOAD_PATH = 'datasets/spectro/'
SAVE_PATH = 'datasets/combined/'

for f in os.listdir(LOAD_PATH):
    if 'LABEL' in f:
        label = np.load(LOAD_PATH + f)
        if 1 in label:
            spectro = np.load(LOAD_PATH + f.split('LABEL')[0]+'.npy')
            combined = np.array((spectro, label))
            np.save(save_path + f.split('LABEL')[0]+'SPEC_LAB.npy', combined)


LOAD_PATH = 'datasets/combined/'
SAVE_PATH = 'datasets/'
SAVE_FILE = '1plus_dataset.npy'
files = os.listdir(LOAD_PATH)

dataset = []
for i in range(len(files)):
    current_file = np.load(LOAD_PATH + files[i])
    dataset.append(current_file)

dataset = np.array(dataset)
np.save(SAVE_PATH + SAVE_FILE, dataset)