import numpy as np

import os
from glob import glob
set_0 = r'C:\Users\USER\Documents\MATLAB\set0'

labels = glob(r'C:\Users\USER\Documents\MATLAB\set_14\labels\Label*.png')
images = glob(r'C:\Users\USER\Documents\MATLAB\set_14\images\image*.png')
labels_dir = r'C:\Users\USER\Documents\MATLAB\set_14\labels\\'
images_dir = r'C:\Users\USER\Documents\MATLAB\set_14\images\\'

print(len(labels), len(images))

def givenumbers(s):
    test = os.path.split(s)[1]
    test = test.replace('_', '.')
    return int(test.split('.')[1])



labels = sorted(labels,key=lambda x: givenumbers(x))
images = sorted(images,key=lambda x: givenumbers(x))
import shutil

for label, image in zip(labels, images):
    image_name = image.replace('.', '\\').split('\\')[-2] + '.png'
    shutil.move(label, labels_dir + image_name)

    print(image_name)
    print()

