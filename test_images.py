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
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import pickle
import os

from model import ModelTreemap
from data_handler import get_testimages

def test_web_images(dataset, ckpt):
    """
        Test images located into the "from_web" folder.
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
    """

    images = []

    # Read all image into the folder
    for filename in os.listdir(r"dataset\test-images"):
        img = Image.open(os.path.join(r"dataset\test-images", filename))
        img = img.resize((48, 48))
        img = np.array(img) / 255
        images.append(img)

    # Load the model
    model = ModelTreemap("Treemap", output_folder=None)
    model.load(ckpt)

    # Get the prediction
    predictions = model.predict(images)

    # Plot the result
    fig, axs = plt.subplots(5, 2, figsize=(10, 25))
    axs = axs.ravel()
    for i in range(10):
        if i%2 == 0:
            axs[i].axis('off')
            axs[i].imshow(images[i // 2])
            axs[i].set_title("Prediction: %s" % np.argmax(predictions[i // 2]))
        else:
            axs[i].bar(np.arange(15), predictions[i // 2])
            axs[i].set_ylabel("Softmax")
            axs[i].set_xlabel("Labels")

    plt.show()


if __name__ == '__main__':
    arguments = docopt(__doc__)
    test_web_images(arguments["<dataset>"], arguments["<ckpt>"])
