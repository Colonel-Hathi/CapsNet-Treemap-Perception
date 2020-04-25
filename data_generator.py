import os
import glob
import pickle
from PIL import Image
import numpy as np
from docopt import docopt

filelist = ['training-data', 'validation-data', 'test-data']

def generate_files():
    base = 'dataset/'
    for file in filelist:
        folder = base + file + '/small'
        labels = os.listdir(folder)
        make_classlabels(labels)
        features = extract_images(folder)
        pairs = {}
        pairs['features'] = features
        pairs['labels'] = labels
        pickle.dump(pairs, open(file + '.p', 'wb'))


def extract_images(folder):
    imagefolder = glob.glob(folder + '/*.png')
    image_list = np.array([np.array(Image.open(images)) for images in imagefolder])
    return image_list


def make_classlabels(list):
    labels = []
    for item in list:
        labels = item.split(".", 1)[0]
    return labels

if __name__ == '__main__':
    generate_files()
    #arguments = docopt(__doc__)
    #generate_files(arguments["<path>"])