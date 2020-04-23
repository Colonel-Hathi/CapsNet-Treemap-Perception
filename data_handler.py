#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle

TRAIN_FILE = "training-data.p"
VALID_FILE = "validation-data.p"
TEST_FILE = "test-data.p"

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
