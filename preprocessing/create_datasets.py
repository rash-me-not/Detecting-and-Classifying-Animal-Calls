import numpy as np
import os


def get_random_idx(features, seed):
    """ Returns random indices with feature size based on a random seed value """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(features.shape[0])
    return indices


def get_data(features, labels_aud, labels_foc, files, indices):
    """Return features, labels and files at given indices """
    features = [features[idx] for idx in indices]
    labels_aud = [labels_aud[idx] for idx in indices]
    labels_foc = [labels_foc[idx] for idx in indices]
    files = [files[idx] for idx in indices]
    return features, labels_aud, labels_foc, files


def get_feature_labels_files(dataset):
    """Returns features, audio_labels, focal_labels and files from a given dataset"""
    features = []
    audio_labels = []
    focal_labels = []
    files = []
    for frame in dataset:
        files.append(frame[0])
        features.append(frame[1][0].T)
        if frame[1][1] is not None:
            audio_labels.append(frame[1][1][0].T)
            focal_labels.append(frame[1][1][1].T)
        else:
            audio_labels.append(None)
            focal_labels.append(None)
    features = np.expand_dims(np.asarray(features), 4)
    audio_labels = np.asarray(audio_labels)
    focal_labels = np.asarray(focal_labels)
    return [features, audio_labels,focal_labels, files]


def get_train_val_indices(dataset, train_ratio, val_ratio):
    """ Based on the given Train and Validation ratio, return random indices from the given dataset"""
    features = get_feature_labels_files(dataset)[0]
    indices = get_random_idx(features, 42).tolist()
    train_indices = indices[:int(train_ratio * len(indices))]
    val_indices = indices[int(train_ratio * len(indices)): int(train_ratio * len(indices)) + int(val_ratio * len(indices)) + 1]
    test_indices = indices[int(train_ratio * len(indices)) + int(val_ratio * len(indices)) + 1::]
    return [train_indices, val_indices, test_indices]

def get_train_val_test(dataset, train_ratio, val_ratio):
    """Return train,test and validation data based on the training and validation ratio. Each dataset that is returned
    contains the features, labels and the filenames"""
    features, labels_aud, labels_foc, files = get_feature_labels_files(dataset)
    train_indices, val_indices, test_indices = get_train_val_indices(dataset, train_ratio, val_ratio)
    x_train, y_train_aud, y_train_foc, train_files = get_data(features, labels_aud, labels_foc, files, train_indices)
    x_val, y_val_aud, y_val_foc, val_files = get_data(features, labels_aud, labels_foc, files, val_indices)
    x_test, y_test_aud, y_test_foc, test_files = get_data(features, labels_aud, labels_foc, files, test_indices)
    return {'train': [x_train, y_train_aud, y_train_foc, train_files], 'test': [x_test, y_test_aud, y_test_foc, test_files], 'val': [x_val, y_val_aud, y_val_foc, val_files]}

def save_data(data, filename, save_path):
    """Saves the dataset in a given file path"""
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, filename), data)