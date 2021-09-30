import cv2
from glob import glob
images = glob(r'C:\Users\USER\PycharmProjects\U-net-with-multiple-classification\data\sperms\*\image\sperm\*.png')

for image in images:
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(image, im)
    print(im.shape)
        #cv2.cvtColor(gray, cv2.CV_GRAY2RGB)

#print(images_dir)
