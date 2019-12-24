import numpy as np
import pandas as pd
import glob
import os
from preprocessing.create_data_groups import fetch_files_with_numcalls


def find_label(file, path):
    """
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
    calls_df  = pd.read_csv(label,
                        sep='\s*\t\s*', header=0
                        )

    if 'Label' in calls_df.columns:
        calls_df.CallType = calls_df.CallType.str[0:3]

        calls_df['Begin Time(t)'] = ((calls_df['StartTime'] - begin_time) * timesteps_per_second).apply(np.floor)
        calls_df['End Time(t)'] = ((calls_df['StartTime']+calls_df['Duration'] - begin_time) * timesteps_per_second).apply(np.floor)
        calls_df = calls_df[calls_df['StartTime'] >= begin_time]
        calls_df = calls_df[(calls_df['StartTime']+calls_df['Duration']) <= end_time]
        if len(calls_df[calls_df['Label'].str.contains('\?')]) > 0:
            calls_df.drop(calls_df.index, inplace=True)
    return calls_df


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
    call_labels = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP','OTH']
    dataframe = dataframe[dataframe['CallType'].isin(call_labels)]
    label = np.zeros((9, timesteps)) # against 8 hyena labels and 1 noise label
    focType = np.zeros((3, timesteps)) # against FOCAL / NON-FOCAL hyena
    label[8,:] = 1
    if 'CallType' in list(dataframe):
        # create update list
        update_list = []
        for index, row in dataframe.iterrows():
            update_list.append([row['Begin Time(t)'],
                                row['End Time(t)'],
                                row['CallType'],
                                row['Focal']])

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
            label[8][begin_t:end_t] = 0

            if l[3] == 0:  # Non Focal
                focType[0][begin_t: end_t] = 1
            elif l[3] == 1:  # Not Defined
                focType[1][begin_t: end_t] = 1
            elif l[3] == 2:  # Focal
                focType[2][begin_t: end_t] = 1

    return [label, focType]

def save_label_file(label_matrix, spec_path, spec_file):
        # Saving the one hot encoded Label file if there is at least one call present in the 6 sec segment
        label_path = os.path.join(spec_path, spec_file + 'LABEL')
        print("Saving file: {}".format(label_path))
        np.save(label_path, label_matrix)
        np.savetxt(label_path, label_matrix, delimiter=",")
