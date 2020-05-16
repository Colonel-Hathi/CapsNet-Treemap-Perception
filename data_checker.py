import os
import glob
import numpy as np
from PIL import Image
from PIL import Image

def main():
    folder = 'dataset/experiments/resolution/'
    path = 'test-data/small/'
    img1 = folder + 'datathesesres48/' + path +'109-0.png'
    img2 = folder + 'datathesesres768/' + path +'22-1.png'
    loop(img1, 'img1')
    loop(img2, 'img2')


def loop(path, name):
    image = Image.open(path)
    resized = image.resize((48, 48), resample=Image.LANCZOS)
    imgarray = np.array(resized)
    image = Image.fromarray(imgarray)
    image.save(name + '.png')


if __name__ == '__main__':
    main()