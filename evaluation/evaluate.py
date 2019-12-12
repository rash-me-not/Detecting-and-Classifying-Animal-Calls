import numpy as np
import pickle
import os
from itertools import chain
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from evaluation.fragments import get_call_ranges, get_fragments, plot_fragments

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

model = load_model('/mar/home/rmishra/PycharmProjects/Detecting-and-Classifying-Animal-Calls/saved_models/model_2019-12-12_08:14:23.198595_network_train/savedmodel.h5')

x_test_aud = np.load("/cache/rmishra/dataset/dataset_aud/x_test.npy")
x_test_acc_ch0 = np.load("/cache/rmishra/dataset/dataset_acc_ch_0/x_test.npy")
x_test_acc_ch1 = np.load("/cache/rmishra/dataset/dataset_acc_ch_1/x_test.npy")
x_test_acc_ch2 = np.load("/cache/rmishra/dataset/dataset_acc_ch_2/x_test.npy")

y_test = np.load('/cache/rmishra/dataset/dataset_aud/y_test.npy')
test_files = np.load('/cache/rmishra/dataset/dataset_aud/test_files.npy')


def IoU(predicted, actual):
    overlap = np.sum(np.logical_and(predicted == 1, actual == 1))
    union = np.sum(np.logical_or(predicted == 1, actual == 1))
    IoU = overlap / union
    return IoU

def plot_confusion_matrix(y_test, preds_w_threshold):
    class_names = {0: 'GIG', 1:'SQL', 2:'GRL', 3:'GRN', 4:'SQT', 5:'MOO', 6:'RUM', 7:'WHP',8:'OTH'}
    preds_w_threshold[preds_w_threshold > 0] = 1
    y_test_ravel = np.argmax(y_test, axis=2).ravel()
    y_pred_ravel = np.argmax(preds_w_threshold, axis=2).ravel()
    cf = confusion_matrix(y_test_ravel, y_pred_ravel, list(class_names.keys()))
    df_cm = DataFrame(cf, index=list(class_names.values()), columns=list(class_names.values()))
    ax = sns.heatmap(df_cm, cmap="Oranges", annot=True, fmt='g')
    ax.set_title("Confusion matrix for label predictions")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('Confusion_matrix.png')
    plt.show()
    plt.close()


preds_w_threshold = model.predict([x_test_aud, x_test_acc_ch0, x_test_acc_ch1, x_test_acc_ch2], batch_size=32)

# Considering noise label as the threshold
for idx, frame in enumerate(preds_w_threshold):
    for row in frame:
        row[:8][row[:8] < row[8]] = 0  # making every sound with less than noise amplitude equal to 0

class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
gt_indices = get_call_ranges(y_test)
pred_indices = get_call_ranges(preds_w_threshold)


# for frame in preds_w_threshold:
#     call_indices = np.unique(np.where(frame[:,:8]!=0)[0])
#     if len(call_indices)>0:
#         indices = np.split(call_indices, np.where(np.diff(call_indices) > 1)[0]+1)
#         for idx_range in indices:
#             max = np.max([frame[val, 8] for val in idx_range])
#             frame[idx_range[0]:idx_range[len(idx_range)-1],:8][frame[idx_range[0]:idx_range[len(idx_range)-1],:8] < max] = 0

for file, gt, pred in zip(test_files, y_test, preds_w_threshold):
    for row_idx in range(259):
        if gt[row_idx, 3]!=0.0 and pred[row_idx, 5]!=0.0:
            truth = gt
            pred_vals = pred
            index = row_idx
            gt_at_idx = gt[row_idx, 5]
            pred_at_idx = pred[row_idx, 3]
            break

plot_confusion_matrix(y_test, preds_w_threshold)
# Performing scoring metrics only on the call labels, and not on the Noise Data.
preds_w_threshold = preds_w_threshold[:, :, :8]
actual = y_test[:, :, :8]

silence_details = get_fragments(y_test, preds_w_threshold, gt_indices)[0]
fragment_details = get_fragments(y_test, preds_w_threshold, gt_indices)[1]


for label_idx in range(fragment_details.shape[1]):
    plot_fragments(fragment_details[:,label_idx], class_names[label_idx])


IoUrecord = []
for i in range(len(x_test_aud)):
    IoUrecord.append(IoU(preds_w_threshold[i], actual[i]))
MeanIoU = sum(IoUrecord) / len(IoUrecord)
print(MeanIoU)

preds_w_threshold[preds_w_threshold > 0] = 1
TP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 1))
FN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 1))
TN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 0))
FP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 0))
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
print("True Positive Rate: " + str(np.round(TPR, 3)))
print("True Negative Rate: " + str(np.round(TNR, 3)))
print("False Positive Rate: " + str(np.round(FPR, 3)))
print("False Negative Rate: " + str(np.round(FNR, 3)))

predicted = model.predict([x_test_aud, x_test_acc_ch0, x_test_acc_ch1, x_test_acc_ch2], batch_size=32)
actual = y_test

fpr, tpr, thresholds = roc_curve(actual.ravel(), predicted.ravel(), pos_label=None)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic Overall')
plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('ROC_overall' + '.png')
plt.show()
plt.close()


for i in range(len(class_names)):
    fpr, tpr, thresholds = roc_curve(actual[:, :, i].ravel(), predicted[:, :, i].ravel(), pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic ' + class_names[i])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('class_ROC_' + class_names[i] + '.png')
    plt.show()
    plt.close()
