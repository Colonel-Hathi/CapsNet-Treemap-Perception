#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Test the model

Usage:
  test.py <ckpt> <dataset>

Options:
  -h --help     Show this help.
  <dataset>     Dataset folder
  <ckpt>        Path to the checkpoints to restore
"""

from docopt import docopt
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt


from model import ModelTreemap
from data_handler import get_images


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(dataset, ckpt):
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """

    # Get Test dataset
    _, _, _, _, X_test, y_test = get_images(dataset)
    X_test = X_test / 255

    model = ModelTreemap("Treemap", output_folder=None)
    # Load the model
    model.load(ckpt)

    #Resize dataset
    print('Test Dataset shape:' + str(X_test.shape))
    print('Test Dataset shape:' + str(y_test.shape))

    # Evaluate all the dataset
    loss, acc, predicted_class = model.evaluate_dataset(X_test, y_test)

    print("Accuracy = ", acc)
    print("Loss = ", loss)

    # Get the confusion matrix
    cnf_matrix = confusion_matrix(y_test, predicted_class)

    # Plot the confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[str(i) for i in range(15)], title='Confusion matrix, without normalization')

    plt.show()


if __name__ == '__main__':
    arguments = docopt(__doc__)
    test(arguments["<dataset>"], arguments["<ckpt>"])
