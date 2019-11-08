import os
import pandas as pd
import shutil


def fetch_files_with_numcalls(base_dir, min_num_of_calls):
    """
    Returns a list of files from the base audio directory with given minimum number of hyena calls

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


def audio_file_save(file_list, file_id, base_dir):
    """
    Copies the desired files with minimum number of calls to a new directory in the form of <base-dir>_<file-id>_converted/audio
    :return:
    """
    save_dir = os.path.abspath(os.path.join(base_dir, '../../', file_id + '_converted', 'audio'))
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    for file in file_list:
        shutil.copy(os.path.join(base_dir, file), os.path.join(save_dir, file))


if __name__ == "__main__":
    base_dir = "/cache/rmishra/cc16_ML/cc16_366a"
    file_list = fetch_files_with_numcalls(base_dir, 1)
    audio_file_save(file_list['audio'], 'cc16_366a', base_dir)
