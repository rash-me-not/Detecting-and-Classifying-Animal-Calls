import numpy as np
import os
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

model = load_model(r'C:\Users\Manav Sanghavi\Projects\PyCharmProjects\Detecting-and-Classifying-Animal-Calls'
                   '\saved_models\model_2019-12-21_21_02_27.534641_network_train\savedmodel.h5')

dataset_dir = "D:\Rashmita\dataset"
x_test_aud = np.load(os.path.join(dataset_dir,"dataset_aud/x_test.npy"))
x_test_acc_ch2 = np.load(os.path.join(dataset_dir,"dataset_acc_ch_2/x_test.npy"))

y_test_aud = np.load(os.path.join(dataset_dir,'dataset_aud/y_test_aud.npy'))
y_test_foc = np.load(os.path.join(dataset_dir,'dataset_aud/y_test_foc.npy'))
test_files = np.load(os.path.join(dataset_dir,'dataset_aud/test_files.npy'))


def IoU(predicted, actual):
    overlap = np.sum(np.logical_and(predicted == 1, actual == 1))
    union = np.sum(np.logical_or(predicted == 1, actual == 1))
    IoU = overlap / union
    return IoU

def getIOU(preds_w_threshold, actual):
    IoUrecord = []
    for i in range(len(x_test_aud)):
        IoUrecord.append(IoU(preds_w_threshold[i], actual[i]))
    return sum(IoUrecord) / len(IoUrecord)


def plot_confusion_matrix(y_test, preds_w_threshold):
    class_names_audio = {0: 'GIG', 1: 'SQL', 2: 'GRL', 3: 'GRN', 4: 'SQT', 5: 'MOO', 6: 'RUM', 7: 'WHP', 8: 'OTH'}
    class_names_foctype = {0: 'NON', 1: 'NOT_DEF', 2: 'FOC'}
    class_names = [class_names_audio, class_names_foctype]
    plt_title = ['Call Type Predictions', 'Focal Type Predictions']

    for i in range(len(preds_w_threshold)):
        preds_w_threshold[i][preds_w_threshold[i] > 0] = 1
        y_test_ravel = np.argmax(y_test[i], axis=2).ravel()
        y_pred_ravel = np.argmax(preds_w_threshold[i], axis=2).ravel()
        cf = confusion_matrix(y_test_ravel, y_pred_ravel, list(class_names[i].keys()))
        df_cm = DataFrame(cf, index=list(class_names[i].values()), columns=list(class_names[i].values()))
        ax = sns.heatmap(df_cm, cmap="Oranges", annot=True, fmt='g')
        ax.set_title("Confusion matrix for " + plt_title[i])
        bottom, top = plt.ylim()
        plt.ylim(bottom + 0.5, top - 0.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('Confusion_matrix_'+plt_title[i]+'.png')
        plt.show()
        plt.close()

def get_tpr_fpr_tnr_fnr(preds_w_threshold, y_test, type):
    TP = np.sum(np.logical_and(preds_w_threshold == 1, y_test == 1))
    FN = np.sum(np.logical_and(preds_w_threshold == 0, y_test == 1))
    TN = np.sum(np.logical_and(preds_w_threshold == 0, y_test == 0))
    FP = np.sum(np.logical_and(preds_w_threshold == 1, y_test == 0))
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    print("True Positive Rate for "+type+": " + str(np.round(TPR, 3)))
    print("True Negative Rate for "+type+": " + str(np.round(TNR, 3)))
    print("False Positive Rate for "+type+": "+ str(np.round(FPR, 3)))
    print("False Negative Rate for "+type+": "+ str(np.round(FNR, 3)))


preds_w_threshold = model.predict([x_test_aud, x_test_acc_ch2], batch_size=32)

class_names = [['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP'], ['NON-FOC','NOTDEF','FOC']]
type = ['Call Type', 'Focal Type']
y_test = [y_test_aud, y_test_foc]

for i in range(len(y_test)):
    fpr, tpr, thresholds = roc_curve(y_test[i].ravel(), preds_w_threshold[i].ravel(), pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic Overall for ' + type[i])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC_overall ' + type[i] + '.png')
    plt.show()
    plt.close()

    for class_idx in range(len(class_names[i])):
        fpr, tpr, thresholds = roc_curve(y_test[i][:, :, class_idx].ravel(), preds_w_threshold[i][:, :, class_idx].ravel(), pos_label=None)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic ' + class_names[i][class_idx])
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('class_ROC_' + class_names[i][class_idx] + '.png')
        plt.show()
        plt.close()


# Considering noise label as the threshold for call label predictions
for idx, frame in enumerate(preds_w_threshold[0]):
    for row in frame:
        row[:8][row[:8] < row[8]] = 0  # making every sound with less than noise amplitude equal to 0

# Considering 0.5 as the threshold for the Focal Type predictions
preds_w_threshold[1][preds_w_threshold[1] > 0.5] = 1
preds_w_threshold[1][preds_w_threshold[1] <= 0.5] = 0

gt_indices = get_call_ranges(y_test_aud)
pred_indices = get_call_ranges(preds_w_threshold[0])

plot_confusion_matrix([y_test_aud, y_test_foc], preds_w_threshold)

# Performing scoring metrics only on the call labels, and not on the Noise Data.
preds_w_threshold[0] = preds_w_threshold[0][:, :, :8]
y_test_aud = y_test_aud[:, :, :8]

MeanIOU_focType = getIOU(preds_w_threshold[1], y_test_foc)
MeanIOU_callType = getIOU(preds_w_threshold[0], y_test_aud)

silence_details = get_fragments(y_test_aud, preds_w_threshold[0], gt_indices)[0]
fragment_details = get_fragments(y_test_aud, preds_w_threshold[0], gt_indices)[1]
for label_idx in range(fragment_details.shape[1]):
    plot_fragments(fragment_details[:,label_idx], class_names[0][label_idx])

get_tpr_fpr_tnr_fnr(preds_w_threshold[0], y_test_aud, 'Call Type')
get_tpr_fpr_tnr_fnr(preds_w_threshold[1], y_test_foc, 'Focal Type')

# for file, gt, pred in zip(test_files, y_test, preds_w_threshold):
#     for row_idx in range(259):
#         if gt[row_idx, 3]!=0.0 and pred[row_idx, 5]!=0.0:
#             truth = gt
#             pred_vals = pred
#             index = row_idx
#             gt_at_idx = gt[row_idx, 5]
#             pred_at_idx = pred[row_idx, 3]
#             break