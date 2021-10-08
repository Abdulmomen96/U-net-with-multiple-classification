from glob import glob
import os
import shutil
images = glob(r'C:\Users\USER\Documents\MATLAB\set_14\labels\image*.png')
images_dir = r'C:\Users\USER\Documents\MATLAB\set_14\labels\\'

def givenumbers(s):
    test = os.path.split(s)[1]
    test = test.replace('_', '.')
    return int(test.split('.')[1])

shift_value = 90

for image in images:
    number = givenumbers(image) + shift_value
    image_name = image.replace('.', '\\').split('\\')[-2][0:6] + str(number) + '.png'
    image_full_name = images_dir + image_name
    print(image, image_full_name)
    shutil.move(image, image_full_name)

print('Done')