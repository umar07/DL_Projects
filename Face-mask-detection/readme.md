# Face Mask Detection
This is a real-time face mask detection project in which it is detected whether the person is wearing a face mask or not. Firstly, we train a deep-learning model to classify images into 2 categories of "With Mask" and "Without Mask". Then this model is saved with all its information(weights,architecture,etc) and later called into the "prediction.py" file for prediction.

In prediction.py we capture the the video using OpenCV library and on every frame that we read, we treat it as an individual image and make prediction on it using our previously trained model. In this way, we are able to classify whether the person is wearing a mask or not in a real-time input feed.

## About the dataset-
This dataset contains about 11,792 equally distributed images of 2 distinct types- WithMask and WithoutMask. 
There are 10,000 training images, 800 validation images, and 992 testing images.

[Dataset Link](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)

## Algorithms used-
* Pre-trained Model- InceptionV3
* Fully connected dense layers for final classification.

## Accuracy result-
evaluated_accuracy = 99%
(The model seems to be overfitted but works fine enough to be used)