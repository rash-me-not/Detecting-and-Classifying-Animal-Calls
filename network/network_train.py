import os
from abc import ABC, abstractmethod

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pickle
from keras.models import load_model
from sklearn.metrics import roc_curve, auc


class NetworkTrain(ABC):
    '''Network Train in declared as an abstract class to train differently for 3axis input or z-axis input'''

    def save_folder_w_datetime(self, date_time):
        date_now = str(date_time.date())
        time_now = str(date_time.time())
        sf = "saved_models/model_" + date_now + "_" + time_now + "_" + os.path.basename(__file__).split('.')[0]
        return sf


    def save_model(self, save_folder, model, model_fit):
        """
        Output: Saves dictionary of model training history as a pickle file.
        """
        model.save(save_folder + '/savedmodel' + '.h5')
        with open(save_folder + '/history.pickle', 'wb') as f:
            pickle.dump(model_fit.history, f)


    def plot_accuracy(self, model_fit, save_folder, history=None):
        """
        Output: Plots and saves graph of accuracy at each epoch.
        """

        train_accuracy_names  = ['output_aud_binary_accuracy', 'output_foctype_binary_accuracy']
        val_accuracy_names  = ['val_output_aud_binary_accuracy','val_output_foctype_binary_accuracy']
        plot_titles = ['Call Types', 'Focal Types']
        (fig, ax) = plt.subplots(2,1, figsize=(8,8))
        for idx, (train_binary_acc, val_binary_acc, plot_title) in enumerate(zip(train_accuracy_names, val_accuracy_names,
                                                                          plot_titles)):
            train_acc = history[train_binary_acc] if history is not None else model_fit.history[train_binary_acc]
            val_acc = history[val_binary_acc] if history is not None else model_fit.history[val_binary_acc]
            epoch_axis = np.arange(1, len(train_acc) + 1)
            ax[idx].set_title('Train vs Validation Accuracy for '+plot_title)
            ax[idx].plot(epoch_axis, train_acc, 'b', label='Train Acc')
            ax[idx].plot(epoch_axis, val_acc, 'r', label='Val Acc')
            ax[idx].set_xlim([1, len(train_acc)])
            ax[idx].set_xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_acc) / 10) + 0.5)))
            ax[idx].legend(loc='lower right')
            ax[idx].set_ylabel('Accuracy')
            ax[idx].set_xlabel('Epochs')
        plt.savefig(save_folder + '/accuracy.png')
        plt.show()
        plt.close()


    def plot_loss(self, model_fit, save_folder, history=None):
        """
        Output: Plots and saves graph of loss at each epoch.
        """

        train_loss_names  = ['output_aud_loss', 'output_foctype_loss']
        val_loss_names  = ['val_output_aud_loss','val_output_foctype_loss']
        plot_titles = ['Call Types', 'Focal Types']
        (fig, ax) = plt.subplots(2,1, figsize=(8,8))
        for idx, (train_op_loss, val_op_loss, plot_title) in enumerate(
                zip(train_loss_names, val_loss_names,
                    plot_titles)):
            train_loss = history[train_op_loss] if history is not None else model_fit.history[train_op_loss]
            val_loss = history[val_op_loss] if history is not None else model_fit.history[val_op_loss]
            epoch_axis = np.arange(1, len(train_loss) + 1)
            ax[idx].set_title('Train vs Validation Loss for '+plot_title)
            ax[idx].plot(epoch_axis, train_loss, 'b', label='Train Loss')
            ax[idx].plot(epoch_axis, val_loss,'r', label='Val Loss')
            ax[idx].set_xlim([1, len(train_loss)])
            ax[idx].set_xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_loss) / 10) + 0.5)))
            ax[idx].legend(loc='upper right')
            ax[idx].set_ylabel('Loss')
            ax[idx].set_xlabel('Epochs')
        plt.savefig(save_folder + '/loss.png')
        plt.show()
        plt.close()


    def plot_ROC(self, model, x, y, save_folder):
        """
        Output: Plots and saves overall ROC graph
        for the validation set.
        """

        predicted = model.predict(x)
        plot_titles = ['Call Types', 'Focal Types']
        (fig, ax) = plt.subplots(2,1, figsize=(8,8))

        for i in range(len(predicted)):
            fpr, tpr, thresholds = roc_curve(y[i].ravel(), predicted[i].ravel(), pos_label=None)
            roc_auc = auc(fpr, tpr)
            ax[i].set_title('Receiver Operating Characteristic Overall for '+plot_titles[i])
            ax[i].plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
            ax[i].legend(loc='lower right')
            ax[i].plot([0,1],[0,1],'r--')
            ax[i].set_xlim([0.0,1.0])
            ax[i].set_ylim([0.0,1.0])
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_xlabel('False Positive Rate')
        plt.savefig(save_folder + '/ROC.png')
        plt.show()
        plt.close()


    def plot_class_ROC(self, model, x_val, y_val, save_folder):
        """
        Output: Plots and saves ROC graphs
        for the validation set.
        """

        class_names = [['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP'], ['NON-FOC', 'NOTDEF', 'FOC']]

        predicted = model.predict(x_val)
        for i in range(len(y_val)):
            for class_idx in range(len(class_names[i])):
                fpr, tpr, thresholds = roc_curve(y_val[i][:, :, class_idx].ravel(),
                                                 predicted[i][:, :, class_idx].ravel(), pos_label=None)
                roc_auc = auc(fpr, tpr)
                plt.title('Receiver Operating Characteristic ' + class_names[i][class_idx])
                plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig(os.path.join(save_folder, 'class_ROC_' + class_names[i][class_idx] + '.png'))
                plt.show()
                plt.close()


    def save_arch(self, model, save_folder):
        with open(save_folder + '/archiecture.txt','w') as f:
        # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def count_labels(self, file):
        count = {}
        sound_label = {0: 'GIG', 1: 'SQL', 2: 'GRL', 3: 'GRN', 4: 'SQT', 5: 'MOO', 6: 'RUM', 7: 'WHP', 8: 'OTH'}
        y = np.load(file)
        data = np.where(y == 1)[2]
        for label_idx in data:
            label = sound_label[label_idx]
            count[label] = count.get(label, 0) + 1
        return count

    def create_save_folder(self, save_folder):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

    @abstractmethod
    def train_network(self):
        pass

    @abstractmethod
    def create_model(self, xtrain, filters, gru_units, dense_neurons, dropout):
        pass

