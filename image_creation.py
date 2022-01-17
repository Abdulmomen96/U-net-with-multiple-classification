import cv2
import numpy as np
import os
import cv2 as cv
images = np.load(r'D:\images0.npy')


number_of_folders = len(images) % 30
next_image = iter(images)
for i in range(number_of_folders):
    os.mkdir('D:/images_raw/set_{}'.format(str(i)))
    image_folder = 'D:/images_raw/set_{}/'.format(str(i))
    for j in range(30):
        image = next(next_image)
        cv2.imwrite(image_folder + "image_" + str(j + 1) + ".jpg",image)


