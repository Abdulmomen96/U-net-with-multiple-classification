import os
from glob import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from load_correct_labels import *
masks = glob(r'.\data\sperms\result\*.png')
images = glob(r'.\data\sperms\test\*.png')
destination = r'.\data\sperms\result\\'
masks = sorted(masks, key= lambda x: int(os.path.basename(x[:-4])))
images = sorted(images, key= lambda x: int(os.path.basename(x[:-4])))
mask_dict = {}
labels_dict = {}
masked_dict = {}
print(len(images))
data = red_csv('counts.csv')
true = []
model_output = []
for image, mask, num in zip(images, masks, range(len(images))):
    original_image = cv2.imread(image)
    mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))
    ret, thresh = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel, iterations=2)
    #cv2.imshow('opening', opening)
    #cv2.waitKey(0)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    #cv2.imshow('sure_bg', sure_bg)
    #cv2.waitKey(0)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #cv2.imshow('dist_transform', sure_bg)
    #cv2.waitKey(0)

    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    #cv2.imshow('sure_fg', sure_fg)
    #cv2.waitKey(0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #cv2.imshow('sure_fg2', sure_fg)
    #cv2.waitKey(0)
    unknown = cv2.subtract(sure_bg, sure_fg)
    #cv2.imshow('unknown', unknown)
    #cv2.waitKey(0)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    count = max(ret, 0)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    print(original_image.shape)
    print(markers.shape)
    markers = cv2.watershed(original_image, markers)
    original_image[markers == -1] = [255, 0, 0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    #cv2.putText(original_image, 'concentration: ' + str(count), (10, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    #cv2.imshow('', original_image)
    print(count)
    #cv2.wait
    #cv2.imwrite(destination +'masked_image_'+ str(num + 1) + '.png', original_image)
    image_name = 'masked_image_'+ str(num + 1) + '.png'
    model_output.append(int(count))
    true.append(int(data[image_name]))
    #print(data[image_name], str(count))

model_output = np.array(model_output)
true = np.array(true)
abs_error = np.abs(np.subtract(model_output, true))
errors = np.divide(abs_error, true + 0.01) * 100


errors = np.sort(errors)
#errors = np.round(errors)
print(errors)
plt.plot(errors,  'r-')
plt.show()
cv2.waitKey(0)
model1 = norm.cdf(errors)
plt.plot(errors, model1,  'r-')
plt.xlim([0, 10])
plt.legend()
plt.title("CDF of Percentage Error")
plt.xlabel('Percentage Error e%')
plt.ylabel('CDF(e)')
plt.grid()
plt.show()
cv2.waitKey(0)

