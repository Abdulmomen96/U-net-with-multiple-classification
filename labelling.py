import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from cv2 import imread, IMREAD_GRAYSCALE
from glob import glob
'''
image_path = glob("E:/CASA_Project/CASA/912/Videos/Data/*.jpg")
print(image_path)




image = imread("E:/CASA_Project/CASA/912/Videos/Data/0a.jpg", IMREAD_GRAYSCALE)
'''

def give_me_con(image):

    image = ~image
    print(image.shape)
    # apply threshold
    thresh = threshold_otsu(image) * 1
    print(thresh)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(image_label_overlay)
    regions_count = 0
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 80:
            regions_count += 1
    return regions_count

'''
for im in image_path:
    image = imread(im, IMREAD_GRAYSCALE)
    con = give_me_con(image)
    print(im)
    print("con:   ", con)
'''
