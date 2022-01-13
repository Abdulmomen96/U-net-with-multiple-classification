
```

python3 main.py -n 001 -lr 0.00004 -ldr 0.000008 -b 16 -s 60 -e 80 -tr 0
```
-lr  = learning rate  
-ldr = learning decay rate  
-b   = batch size  
-s   = steps  
-e   = epochs
-tr = training 0, testing 1
-vd = videos input (For motility)
(check more params in mode/config.py)


--------------------------------------------------------------------------------


### You have to know:
The structure of this project is:

/data/sperms/train/image/sperm : The data of the training images image_0.png in this naming format.
/data/sperms/train/label/sperm/ : The data of the training labels or Masks image_0.png in this naming format.

/data/sperms/test/sperm/ : Testing images named from 1 to n where n is passed with other arguments or passed as 27 by default.,

/data/sperms/result/ : Results directory to save the genrated images from the network in the testing.

incase there are more than one label are introduce i.e, sperm, nonsperm, there should be a new directory with that label with the dataset similar to the sperm

All you have see are defined as below:
* data.py : prepare the related images you want to train and predict.
* model.py : define the U-net structure
* main.py : run the program

Supplementry files
gray_2_rgb.py
jpg_to_png.py
              The first two files, as the names sugest, they are used to preprocess the images, it really depends on how you prepared your dataset,
              For the training you need a png grayscale images.
rename_shift.py
renaming_gt_labels.py
renaming_test_images.py
              These files are used to rename the training and testing images/labels, in an accptible way as stated above.

### data.py
This file is responsible for the preprocessing of the dataset, using the following functions:
adjustData() normalize the training images and the labels, and also maps the label colored image into a multi-layer depending on the number of labels. Since we only used only  one label it's ok to use gray images. 

trainGenerator(), testGenerator()  generates the training and testing dataset, and also accepts some paramters which controls the data augmentation in order to increase the dataset size.


saveResult() converts the resulted array from the model into images and saves the results into the result folder /data/sperms/result/


### model.py

All the U-net architecture is defined in model.py.


### main.py

Training and test steps are defined in main.py, whcih are selected by passing the tr argument when running main.py file.


### dependencies

* Tensorflow : 1.4.0
* Keras >= 1.0
* Python 3.5.2


 
Preparing the dataset:
To prepare the dataset we used image labeler tool. However, the output of the image labeler tool can't be used directly since the labels are mapped from 1 to 255, which makes them look blank in case of using a small number of labels. We used exporting_images.m to solve this.


For motility, the saved model for concentration is used for the testing. the viedos are stored in the test directory, and fed to the model to generate the output images.
To commine the output images of the sperms and draw the path traveled by each sperm we run the distance.py file. This file filters and smoothes the highlited sperm shapes in the ouput images and then combines then into single image to draw paths for each spearm.

For working with the code please watch the attatched video.
  

