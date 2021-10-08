from PIL import Image
from glob import glob
images_dir = glob(r'C:\Users\USER\Documents\MATLAB\set_14\images\image*.jpg')

for im in images_dir:
    name = im[0:-3]+'png'
    image_temp = Image.open(im)
    image_temp.save(name)
