import cv2
from glob import glob
import numpy as np
from PIL import Image
def get_num(s):
  if 'txt' in s:
    return 1000
  return int(s.replace('\\', '.').split('.')[-2])
sequence = glob('./data/sperms/test/Video_results/*.png')
print(sequence)
sequence.sort(key=lambda s: get_num(s))

start = np.zeros((256, 256, 3))

frames = []
for img in sequence:
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

    start = (temp + start) % 256
    frames.append(np.array(start))
    cv2.imshow('image', start)
    cv2.waitKey(0)
cv2.imwrite('result.png', start)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, 70, (640, 480))
for image in frames:
    frame = cv2.resize(image, (640, 480))
    out.write(frame)

out.release()
