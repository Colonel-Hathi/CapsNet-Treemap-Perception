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
        features = extract_images(folder)
        #labeldata = labels
        #zipdata = zip(imagedata, labeldata)
        pairs = {}
        pairs['features'] = features
        pairs['labels'] = labels
        pickle.dump(pairs, open(file + '.p', 'wb'))


def extract_images(folder):
    image_list = []
    imagefolder = folder + '/*.png'
    for images in glob.glob(imagefolder):
        image = Image.open(images)
        image_list.append(np.asarray(image))
    return np.array(image_list)


if __name__ == '__main__':
    generate_files()
    #arguments = docopt(__doc__)
    #generate_files(arguments["<path>"])