import cv2
import numpy as np
import os

#function to exclude non avi vido files
def is_jpg(s):
    if s.find(".jpg") > 0:
        return True
    else:
        return False

# should be changed according to the samples' directory
samples_directorys = [os.path.abspath(r"E:\CASA_Project\CASA"), os.path.abspath(r"E:\CASA_Project\CASA2"), os.path.abspath(r"E:\CASA_Project\CASA3")]

i = 0
for samples_directory in samples_directorys:
    # sub-directories listing
    subdir = [f.path for f in os.scandir(samples_directory) if f.is_dir()]
    images_paths = []  # stores images' paths
    images = []  # actual images
    labels = []  # concentrations

    # image extraction
    for dir in subdir:
        print(dir)
        label_file = os.path.join(dir, "new_concentration.txt")
        label = 0
        with open(label_file, 'r') as f:
            label = float(f.readline().strip("\n\r"))
            print(label)
        os.chdir(os.path.join(dir, "Videos/Data"))
        images_paths.append([os.path.join(dir, "Videos/Data", f) for f in os.listdir(os.path.join(dir, "Videos/Data")) if is_jpg(f)])
        for image_path in images_paths[-1]:
            images.append(cv2.imread(image_path, 0))
            labels.append(label)
    print(len(images))
    print(len(labels))
    #cv2.imshow('',images[-1])
    print(labels[-1])
    #cv2.waitKey(0)

    np.save(r'E:\CASA_Project\Dataset\images{}.npy'.format(str(i)), np.array(images))
    np.save(r'E:\CASA_Project\Dataset\labels{}.npy'.format(str(i)), np.array(labels))
    i += 1
    cv2.destroyAllWindows()
