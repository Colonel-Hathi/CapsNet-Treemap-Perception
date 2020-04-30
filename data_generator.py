import os
import glob
import pickle
from PIL import Image
import numpy as np

# List of pickle files to generate
filelist = ['training-data', 'validation-data', 'test-data']

# Generate pickle files containing dict with 4D array of image features and 1D array of labels
def generate_files():
    # Dataset folder
    base = 'dataset/Minidata/'
    for file in filelist:
        # Image folder
        folder = base + file + '/small'
        # Generate label list from directory
        labels = os.listdir(folder)
        labels = make_classlabels(labels)
        features = extract_images(folder)
        # Generate and fill dictionary with key/value pairs 'features' and 'labels'
        pairs = {}
        pairs['features'] = features
        pairs['labels'] = labels
        # Write dict to pickle file
        pickle.dump(pairs, open('m' + file + '.p', 'wb'))

# Extract images from folder and convert to 4D numpy array
def extract_images(folder):
    imagefolder = glob.glob(folder + '/*.png')
    image_list = np.array([np.array(Image.open(images)) for images in imagefolder])
    return image_list

# Round label names to whole numbers
def make_classlabels(list):
    labels = []
    for item in list:
        labels = item.split(".", 1)[0]
    return labels

if __name__ == '__main__':
    generate_files()