# Convolutional neural network Networks Implementation from scratch

#### Training on custom dataset

## Dataset format:
#### Root directory
* The root directory should contain all the image files related to the dataset.
#### CSV File 
* The dataset is accompanied by a CSV file that contains two columns: "Image Name" and "Class". This CSV file serves as a reference to associate each image with its respective class or label.
* The "Image Name" column contains the names of the image files present in the root directory. Each entry in this column should uniquely identify an image file.
* The "Class" column represents the corresponding class or label for each image. It contains numerical values.

## LeNet

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/08e82682-092e-4266-9be4-51511c859637)

Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

* LeNet uses tanh and sigmoid activation function.
* Input image size: 32*32
