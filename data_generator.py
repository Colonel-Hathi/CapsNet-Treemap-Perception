import os
import glob
import pickle
from PIL import Image
from docopt import docopt


class FileData(object):
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fileobj:
            self.data = fileobj.read()


def generate_files(path):
    base = 'dataset/'
    folder = base + path
    labels = os.listdir(folder)
    images = extract_images(folder)
    pairs = dict(zip(images, labels))
    pickle.dump(pairs, open(path + '.p', 'wb'))


def extract_images(folder):
    image_list = []
    imagefolder = folder + '/*.png'
    for images in glob.glob(imagefolder):
        im = Image.open(images)
        image_list.append(im)
    return image_list


if __name__ == '__main__':
    generate_files('training-data')
    #arguments = docopt(__doc__)
    #generate_files(arguments["<path>"])