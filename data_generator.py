import os
import glob
import pickle
from PIL import Image
import numpy as np
from docopt import docopt


def generate_files(path):
    base = 'dataset/'
    folder = base + path + '/large'
    labels = os.listdir(folder)
    imagedata = extract_images(folder)
    labeldata = str(labels)
    pairs = dict(zip(imagedata, labeldata))
    pickle.dump(pairs, open(path + '.p', 'wb'))


def extract_images(folder):
    image_list = []
    imagefolder = folder + '/*.png'
    for images in glob.glob(imagefolder):
        imagedata = load_image(images)
        image_list.append(imagedata)
    return tuple(image_list)
    #return str(image_list)


def load_image(imagepath):
    image = Image.open(imagepath)
    #image.load()
    imagedata = np.array(image)
    #imagedata = list(image.getdata())
    data = str(imagedata)
    return data


if __name__ == '__main__':
    generate_files('test-data')
    #arguments = docopt(__doc__)
    #generate_files(arguments["<path>"])