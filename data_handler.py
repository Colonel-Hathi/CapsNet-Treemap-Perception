#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from PIL import Image
import pickle
from PIL import Image
import tensorflow as tf

## small (96x96) images or large (300x300) images
IMAGE_SIZE = 'small'

def get_images(folder):
    filelist = ['training-data', 'validation-data', 'test-data']
    base = folder
    for file in filelist:
        # Image folder
        imagefolder = base + file + IMAGE_SIZE
        # Generate label list from directory
        labels = os.listdir(imagefolder)
        labels = make_classlabels(labels)
        features = extract_images(imagefolder)
        # Generate and fill dictionary with key/value pairs 'features' and 'labels'
        pairs = {}
        pairs['features'] = features
        pairs['labels'] = np.array(labels, dtype=np.uint8)
        if file == 'training-data':
            train = pairs
        if file == 'validation-data':
            valid = pairs
        if file == 'test-data':
            test = pairs

    # Retrieve all data
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_testimages(folder):
    # Image folder
    imagefolder = folder + '/test-data/small'
    # Generate label list from directory
    labels = os.listdir(imagefolder)
    labels = make_classlabels(labels)
    features = extract_images(imagefolder)
    # Generate and fill dictionary with key/value pairs 'features' and 'labels'
    pairs = {}
    pairs['features'] = features
    pairs['labels'] = np.array(labels, dtype=np.uint8)
    test = pairs

    # Retrieve data
    X_test, y_test = test['features'], test['labels']

    return X_test, y_test


def extract_images(folder):
    imagefolder = glob.glob(folder + '/*.png')
    image_list = []
    for images in imagefolder:
        image = Image.open(images)
        resizeimg = image.resize((48, 48), resample=Image.LANCZOS)
        imgarray = np.array(resizeimg)
        image_list.append(imgarray)
    #print('IMAGE LIST SHAPE:' + str(image_list.shape))
    return np.array(image_list)


def make_classlabels(list):
    labels = []
    for item in list:
        label = item.split("-", 1)[1]
        label = label.split(".", 1)[0]
        labels.append(label)
    return labels