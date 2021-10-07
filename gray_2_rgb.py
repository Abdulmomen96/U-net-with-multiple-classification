import cv2
from PIL import Image
from glob import glob
images = glob(r'C:\Users\USER\PycharmProjects\U-net-with-multiple-classification\data\sperms\train\*\sperm\*.png')

for image in images:
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    #im1 = Image.open(image)
    #print(image[0:-3])
    #im1.save(image[0:-3] + 'png')
    cv2.imwrite(image, im)
    print(im.shape)
        #cv2.cvtColor(gray, cv2.CV_GRAY2RGB)

#print(images_dir)
