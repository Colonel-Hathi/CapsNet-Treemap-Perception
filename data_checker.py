import os
import glob
import numpy as np
from PIL import Image
from PIL import Image

def main():
    folder = 'dataset/nodedata/training-data/small/'
    imgname = '113-0.png'
    path = folder + imgname
    image = Image.open(path)
    resized = image.resize((48, 48), resample=Image.LANCZOS)
    imgarray = np.array(resized)
    show = Image.fromarray(imgarray)
    show.save('image.png')
    show.show()
    print('Image printed')


if __name__ == '__main__':
    main()