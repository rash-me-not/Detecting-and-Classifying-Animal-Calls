import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# def create_train_test(base_dir):
#     df = pd.DataFrame(columns=['file_id', 'fname', 'type'])
#     for file in os.listdir(base_dir):
#         if os.path.splitext(file)[-1].lower() == ".npy":
#             file_id = file.split(".")[0][:-5] if "LABEL" in file else file.split(".")[0]
#             type = "LABEL" if "LABEL" in file else "SPEC"
#             df.loc[len(df)] = [file_id, file, type]
#
#     df = df.pivot(index="file_id", columns="type", values="fname")
#
#     features = []
#     labels = []
#     longest_seq = -1
#     for spec_file, label_file in zip(df['SPEC'], df['LABEL']):
#         print("Adding feature and label for: " + spec_file)
#         spec = np.load(os.path.join(base_dir, spec_file)).T
#         label = np.load(os.path.join(base_dir, label_file)).T
#         longest_seq =  max(longest_seq, spec.shape[0])
#         features.append(np.lib.pad(spec, ((0, longest_seq - len(spec)), (0, 0)), 'constant', constant_values=0))
#         labels.append(np.lib.pad(label, ((0, longest_seq - len(label)), (0, 0)), 'constant', constant_values=0))
#
#     return [np.asarray(features), np.asarray(labels)]


base_dir = '/cache/rmishra/cc16_366a_converted'
SAVE_PATH = os.path.join(base_dir,'datasets')
#
#
# x, y = create_train_test(base_dir+'/spectro')
# x = np.expand_dims(x, 4)

dataset = np.load('/cache/rmishra/cc16_366a_converted/datasets/1plus_dataset.npy')
x = dataset[:, 0]
y = dataset[:, 1]
# reformat x to (n, timesteps, mel bands, 1)
features = []
labels = []
longest_seq = -1
for spec, label in zip(x, y):
    spec = spec.T
    label= label.T
    longest_seq =  max(longest_seq, spec.shape[0])
    features.append(np.lib.pad(spec, ((0, longest_seq - len(spec)), (0, 0)), 'constant', constant_values=0))
    labels.append(np.lib.pad(label, ((0, longest_seq - len(label)), (0, 0)), 'constant', constant_values=0))

features = np.expand_dims(np.asarray(features),4)
labels = np.asarray(labels)

# split data into test, train and validation
TRAIN = 0.75
VALIDATION = 0.15
TEST = 0.10

# create train set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1 - TRAIN)

# create val and test set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=TEST/(TEST + VALIDATION)) 

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# save files
np.save(os.path.join(SAVE_PATH, 'x_train.npy'), x_train)
np.save(os.path.join(SAVE_PATH, 'x_val.npy'), x_val)
np.save(os.path.join(SAVE_PATH, 'x_test.npy'), x_test)
np.save(os.path.join(SAVE_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(SAVE_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y_test)