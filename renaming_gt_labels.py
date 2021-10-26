import numpy as np

import os
from glob import glob
set_0 = r'C:\Users\USER\Documents\MATLAB\set0'

labels = glob(r'C:\Users\USER\Downloads\data2\csa\label\\*')
images = glob(r'C:\Users\USER\Downloads\data2\csa\image\\*')
labels_dir = r'C:\Users\USER\Downloads\data2\csa\label\\'
images_dir = r'C:\Users\USER\Downloads\data2\csa\image\\'

print(len(labels), len(images))

def givenumbers(s):
    test = os.path.split(s)[1]
    test = test.replace('_', '.')
    return int(test.split('.')[1])



labels = sorted(labels,key=lambda x: givenumbers(x))
images = sorted(images,key=lambda x: givenumbers(x))
import shutil

i = 0
for label, image in zip(labels, images):
    image_name = str(i) + '.png' #image.replace('.', '\\').split('\\')[-2] + '.png'
    shutil.move(label, labels_dir + image_name)
    shutil.move(image, images_dir + image_name)
    i += 1

    print(image_name)
    print()

