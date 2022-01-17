import os
import cv2
from labelling import give_me_con
from tqdm import tqdm
#function to exclude non avi vido files
def is_avi(s):
    if s.find(".avi") > 0:
        return True
    else:
        return False

#should be changed according to the samples' directory

samples_directorys = [os.path.abspath(r"E:\CASA_Project\CASA")] #os.path.abspath(r"E:\CASA_Project\CASA2"),


for samples_directory in samples_directorys:
    #sub-directories listing
    subdir = [f.path for f in os.scandir(samples_directory) if f.is_dir()]
    videos = [] #stores videos' paths

    #image extraction
    for dir in tqdm(subdir):
        print(dir)
        os.chdir(os.path.join(dir, "Videos"))
        if not os.path.exists("./Data"):
            os.mkdir("./Data")
        videos.append([os.path.join(dir, "Videos", f) for f in os.listdir(os.path.join(dir, "Videos")) if is_avi(f)])
        current_video = 0  # video number
        concentration = 0
        if len(videos[-1]) < 1:
            continue
        for vid in videos[-1]:
            #reading the video
            cam = cv2.VideoCapture(vid)
            frames = []
            current_frame = 0

            #iterating through frames
            while (True):
                ret, frame = cam.read()
                if ret:
                    frames.append(frame)
                    current_frame += 1
                else:
                    print(current_frame)
                    #taking 3 frames from each video
                    if len(frames) < 5:
                        break
                    cv2.imwrite('./Data/' + str(current_video) + "a.jpg", frames[int(current_frame * 1 / 5)])
                    cv2.imwrite('./Data/' + str(current_video) + "b.jpg", frames[int(current_frame * 1 / 2)])
                    cv2.imwrite('./Data/' + str(current_video) + "c.jpg", frames[int(current_frame * 4 / 5)])
                    image0 = cv2.cvtColor(frames[int(current_frame * 1 / 5)], cv2.COLOR_RGB2GRAY)
                    image1 = cv2.cvtColor(frames[int(current_frame * 1 / 2)], cv2.COLOR_RGB2GRAY)
                    image2 = cv2.cvtColor(frames[int(current_frame * 4 / 5)], cv2.COLOR_RGB2GRAY)

                    concentration += give_me_con(image0)
                    concentration += give_me_con(image1)
                    concentration += give_me_con(image2)
                    break
            current_video += 1
            cam.release()
        concentration /= (len(videos[-1]) * 3)
        open("../new_concentration.txt", 'w').close()
        with open("../new_concentration.txt", 'w') as f:
            f.write("{:3.2f}".format(concentration))
    cv2.destroyAllWindows()
