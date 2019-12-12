import os
import pandas as pd
import shutil


def fetch_files_with_numcalls(base_dir, min_num_of_calls):
    """
    Returns a list of files from the base audio directory with given minimum number of hyena calls
    The result contains the list of file_identifiers, along with their values for different file types
    In our project, we have 4 file types associated with every id, i.e. the accelerometer data, audio data, label file and the call file
    Example: file id cc16_352a_14401s has value cc16_352a_14401s_acc.wav, cc16_352a_14401s_audio.wav,
    cc16_352a_14401s_calls.txt and cc16_352a_14401s_labels.txt
    """
    file_list = pd.DataFrame(columns=['file_identifier', 'filename', 'file_type'])
    for dir, _subdirs, files in os.walk(base_dir):
        for fname in files:
            try:
                [file_id, file_type] = fname.rsplit(sep="_", maxsplit=1)
                file_type = file_type.split(".")[0]
                file_list.loc[len(file_list)] = [file_id, fname, file_type]
            except:
                print(fname + "cannot be split into 2")
    file_list = file_list.pivot(index='file_identifier', columns='file_type', values='filename')
    file_list['num_of_calls'] = file_list.apply(lambda row: len(open(base_dir + '/' + row.labels, 'rb').readlines()), axis=1)
    return file_list[file_list['num_of_calls'] >= min_num_of_calls]


