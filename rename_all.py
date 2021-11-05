from glob import glob
import cv2

images = glob(r'D:\images_raw\set_*\*.jpg')
destination = r'D:\images\\'
for i, image in enumerate(images):
    pixels = cv2.imread(image)
    cv2.imwrite(destination + str(i + 1) + '.png', pixels)
    print(str(i + 1), len(images))
