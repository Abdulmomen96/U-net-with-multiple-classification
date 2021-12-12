import cv2
from glob import glob
import numpy as np
from PIL import Image
def get_num(s):
  if 'txt' in s:
    return 1000
  return int(s.replace('\\', '.').split('.')[-2])
sequence = glob('./data/sperms/test/video_results/*.png')
print(sequence)
sequence.sort(key=lambda s: get_num(s))

start = np.zeros((256, 256, 3))
first_image = np.zeros((256, 256, 3))
frames = []
for img, i in zip(sequence, range(len(sequence))):
    temp = np.zeros((256, 256, 3))
    img = cv2.imread(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for thresh_val in range(0, 255):

        # threshold and detect contours
        thresh = cv2.threshold(imgray, thresh_val, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh,
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        # filter contours by area
        min_area = 20
        max_area = 70
        print(cv2.contourArea(contours[0]))
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area and cv2.contourArea(c) <= max_area]
        area_contours = [cv2.contourArea(c) for c in filtered_contours]

        # acceptable deviation from median contour area
        median_area = np.median(area_contours)
        dev = 0.3
        lowerb = median_area - dev * median_area
        upperb = median_area + dev * median_area

        # break when all contours are within deviation from median area
        if ((area_contours > lowerb) & (area_contours < upperb)).all():
            break

    # draw center location of blobs
    line_length = 8
    cross_color = (255, 0, 0)
    for c in filtered_contours:
        M = cv2.moments(c)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        cv2.circle(temp, (x, y), 1, cross_color, 2)
        #cv2.line(img, (x - line_length, y), (x + line_length, y), cross_color, 2)
        #cv2.line(img, (x, y - line_length), (x, y + line_length), cross_color, 2)

    if i == 0:
        first_image = start

    start = (temp + start) % 256
    #cv2.imshow('image', start)
    #cv2.waitKey(0)
    frames.append(np.array(start))

cv2.imshow('image', start)
cv2.waitKey(0)
cv2.imshow('image', start - first_image)
cv2.waitKey(0)
cv2.imwrite('result.png', start)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, 70, (640, 480))
for image in frames:
    frame = cv2.resize(image, (640, 480))
    out.write(frame)

out.release()

start = cv2.imread('result.png', cv2.IMREAD_GRAYSCALE)
start = np.ubyte((start / np.max(start)) * 255)

#start = cv2.cvtColor(start, cv2.COLOR_RGB2GRAY)
cv2.imshow('thresh', start)
cv2.waitKey(0)
original_image = start
mask_image = start
#mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))
ret, thresh = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow('opening', opening)
cv2.waitKey(0)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=1)
cv2.imshow('sure_bg', sure_bg)
cv2.waitKey(0)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
cv2.imshow('dist_transform', sure_bg)
cv2.waitKey(0)

ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

cv2.imshow('sure_fg', sure_fg)
cv2.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
cv2.imshow('sure_fg2', sure_fg)
cv2.waitKey(0)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('unknown', unknown)
cv2.waitKey(0)
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
#font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#cv2.putText(original_image, 'concentration: ' + str(count), (10, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow('', original_image)
print(count)