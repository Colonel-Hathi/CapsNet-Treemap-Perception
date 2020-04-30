#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from PIL import Image


def get_images(folder):
    filelist = ['training-data', 'validation-data', 'test-data']
    base = folder #+ 'minidata/'
    for file in filelist:
        # Image folder
        imagefolder = base + file + '/small'
        # Generate label list from directory
        labels = os.listdir(imagefolder)
        labels = make_classlabels(labels)
        features = extract_images(imagefolder)
        # Generate and fill dictionary with key/value pairs 'features' and 'labels'
        pairs = {}
        pairs['features'] = features
        pairs['labels'] = labels
        if file == 'training-data':
            train = pairs
        if file == 'validation-data':
            valid = pairs
        if file == 'test-data':
            test = pairs
            write_contents_txt(test)

    # Retrieve all data
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def write_contents_txt(file):
    f = open('file.txt', 'w')
    f.write(str(file))
    f.close


def extract_images(folder):
    imagefolder = glob.glob(folder + '/*.png')
    #image_list = np.array([np.array(Image.open(images)) for images in imagefolder])
    image_list = []
    for images in imagefolder:
        image = np.array(Image.open(images))
        image_list.append(image)
    image_list = np.array(image_list)
    #print('IMAGE LIST SHAPE:' + str(image_list.shape))
    return image_list


def make_classlabels(list):
    labels = []
    for item in list:
        label = item.split(".", 1)[0]
        labels.append(label)
    return labels


TRAIN_FILE = "mtraining-data.p"
VALID_FILE = "mvalidation-data.p"
TEST_FILE = "mtest-data.p"

def get_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    # Load the dataset
    #training_file = os.path.join(folder, TRAIN_FILE)
    #validation_file = os.path.join(folder, VALID_FILE)
    #testing_file = os.path.join(folder, TEST_FILE)

    with open(TRAIN_FILE, mode='rb') as f:
        train = pickle.load(f)
    with open(VALID_FILE, mode='rb') as f:
        valid = pickle.load(f)
    with open(TEST_FILE, mode='rb') as f:
        test = pickle.load(f)

    # Retrieve all data
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test
