import numpy as np
import pandas as pd
import glob
import os
from create_data_groups import fetch_files_with_numcalls


def find_label(file, path):
    """
    ***OLD:
    Output: Audit label file relating to input spectro file
    There are a few instances where a file has multiple audit label files:
    cc352a021_6h
    cc352b003_4h
    cc354a049_8h
    in these cases the first label will be used.]

    ***NEW:
    Retrieving the label text file from the data set against a given file id
    example: Return cc16_366a_344322s_labels.txt for audio file cc16_366a_344322s_audio.wav
    """
    label_search = file.rsplit("_",maxsplit=1)[0]
    label = fetch_files_with_numcalls(path,1).loc[label_search]['labels']
    return label


def create_label_dataframe(label, begin_time, end_time, window_size, timesteps_per_second):
    """
    Output: Dataframe with relevant labels for the
    spectrogram file
    """
    labels_df = pd.read_csv(label,
                            sep='\t',
                            names=['StartTime','Duration','Label'])
                            # index_col='Selection')
    if 'Label' in labels_df.columns:
        # filter for any labels that do not start with definitive call type label
        # call_labels = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
        # change labels to first 3 characters
        labels_df.Label = labels_df.Label.str[0:3]
        # labels_df = labels_df[labels_df['Label'].isin(call_labels)]
        labels_df['Begin Time(t)'] = ((labels_df['StartTime'] - begin_time) * timesteps_per_second).apply(np.floor)

        labels_df['End Time(t)'] = ((labels_df['StartTime']+labels_df['Duration'] - begin_time) * timesteps_per_second).apply(np.floor)
        labels_df = labels_df[labels_df['StartTime'] >= begin_time]
        labels_df = labels_df[(labels_df['StartTime']+labels_df['Duration']) <= end_time]
        if len(labels_df[labels_df['Label'].str.contains('\?')]) > 0:
            labels_df.drop(labels_df.index, inplace=True)
    return labels_df


def create_label_matrix(dataframe, timesteps):
    """
    Output: Matrix of 0s and 1s. Each column represents a timestep,
    Each row represents a different call type:
    Row 0 = Giggle (GIG)
    Row 1 = Squeal (SQL)
    Row 2 = Growl (GRL)
    Row 3 = Groan (GRN)
    Row 4 = Squitter (SQT)
    Row 5 = Low / Moo (MOO)
    Row 6 = Alarm rumble (RUM)
    Row 7 =  Whoop (WHP)
    For example:
    [[0, 0, 0, 0, 0, 0 ....],
    [0, 0, 0, 0, 0, 0 ....],
    [0, 0, 0, 1, 1, 1 ....], This represents a Growl in 3-5 timesteps
    [0, 0, 0, 0, 0, 0 ....],
    [0, 0, 0, 0, 0, 0 ....],
    [0, 0, 0, 0, 0, 0 ....],
    [0, 0, 0, 0, 0, 0 ....],
    [1, 1, 1, 1, 0, 0 ....],] This represents a Whoop in 0-3 timesteps
    """
    call_labels = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
    dataframe = dataframe[dataframe['Label'].isin(call_labels)]
    label = np.zeros((8, timesteps))
    if 'Label' in list(dataframe):
        # create update list
        update_list = []
        for index, row in dataframe.iterrows():
            update_list.append([row['Begin Time(t)'],
                                row['End Time(t)'],
                                row['Label']])
        # label correct row based on label
        for l in update_list:
            begin_t = int(l[0])
            end_t = int(l[1]) + 1
            if l[2] == 'GIG':
                label[0][begin_t:end_t] = 1
            elif l[2] == 'SQL':
                label[1][begin_t:end_t] = 1
            elif l[2] == 'GRL':
                label[2][begin_t:end_t] = 1
            elif l[2] == 'GRN':
                label[3][begin_t:end_t] = 1
            elif l[2] == 'SQT':
                label[4][begin_t:end_t] = 1
            elif l[2] == 'MOO':
                label[5][begin_t:end_t] = 1
            elif l[2] == 'RUM':
                label[6][begin_t:end_t] = 1
            elif l[2] == 'WHP':
                label[7][begin_t:end_t] = 1
    return label


paths = ['/cache/rmishra/cc16_366a_converted/spectro/']
base_dir = "/cache/rmishra/cc16_ML/cc16_366a"



for path in paths:
    if(os.path.exists(path)):
        for f in os.listdir(path):
            if 'LABEL' not in f:
                label = find_label(f, base_dir)           #We dont have different duration files
                begin_time = int(f.split('_')[-1].split('sto')[0])
                end_time = int(f.split('_')[-1].split('sto')[1].split('s')[0])
                window_size = end_time - begin_time
                timesteps = 259  # need to set timesteps
                timesteps_per_second = timesteps / window_size
                df = create_label_dataframe(os.path.join(base_dir,label),          #changing first parameter from label to f
                                            begin_time,
                                            end_time,
                                            window_size,
                                            timesteps_per_second)
                label_matrix = create_label_matrix(df, timesteps)
                print("Saving file: {}".format(path + f[:-4] + 'LABEL'))
                np.save(path + f[:-4] + 'LABEL', label_matrix)
                np.savetxt(path + f[:-4] + 'LABEL', label_matrix, delimiter=",")
