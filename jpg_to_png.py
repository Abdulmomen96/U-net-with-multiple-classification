from PIL import Image
from glob import glob
images_dir = glob(r'D:\data2\csa\image\\*')
image_temp = Image.open(images_dir)

for im in images_dir:
    name = im[0:-3]+'png'
    #image_temp = Image.open(im)
    print("Hi")
    #image_temp.save(name)
