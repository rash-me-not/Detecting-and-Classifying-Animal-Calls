
import numpy as np
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)

model = load_model('../network/saved_models/model_2019-11-08_02:15:30.247730_network_train/savedmodel.h5')
x_test = np.load('/cache/rmishra/cc16_366a_converted/datasets/x_test.npy')
y_test = np.load('/cache/rmishra/cc16_366a_converted/datasets/y_test.npy')


def IoU(predicted, actual):
    overlap = np.sum(np.logical_and(predicted == 1, actual == 1))
    union = np.sum(np.logical_or(predicted == 1, actual == 1))
    IoU = overlap/union
    return IoU


threshold = 0.5
preds_w_threshold = model.predict(x_test)
preds_w_threshold[preds_w_threshold > threshold] = 1
preds_w_threshold[preds_w_threshold <= threshold] = 0
actual = y_test


IoUrecord = []
for i in range(len(x_test)):
    IoUrecord.append(IoU(preds_w_threshold[i], actual[i]))
MeanIoU = sum(IoUrecord)/len(IoUrecord)

TP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 1))
FN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 1))
TN = np.sum(np.logical_and(preds_w_threshold == 0, actual == 0))
FP = np.sum(np.logical_and(preds_w_threshold == 1, actual == 0))
TPR  = TP / (TP + FN)
TNR  = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
print("True Positive Rate: " + str(np.round(TPR, 3)))
print("True Negative Rate: " + str(np.round(TNR, 3)))
print("False Positive Rate: " + str(np.round(FPR, 3)))
print("False Negative Rate: " + str(np.round(FNR, 3)))


predicted = model.predict(x_test)
actual = y_test


fpr, tpr, thresholds = roc_curve(actual.ravel(), predicted.ravel(), pos_label=None)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic Overall')
plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.close()



class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
for i in range(len(class_names)):
    fpr, tpr, thresholds = roc_curve(actual[:,:,i].ravel(), predicted[:,:,i].ravel(), pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic ' + class_names[i])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()
