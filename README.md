# Convolutional Neural Network Implementation from scratch

## Dataset format:
#### Root directory
* The root directory should contain all the image files related to the dataset.
#### CSV File 
* The dataset is accompanied by a CSV file that contains two columns: "Image Name" and "Class". This CSV file serves as a reference to associate each image with its respective class or label.
* The "Image Name" column contains the names of the image files present in the root directory. Each entry in this column should uniquely identify an image file.
* The "Class" column represents the corresponding class or label for each image. It contains numerical values.

#### Note: Custom Dataset will be used to train the networks present in this repository.

## LeNet

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/08e82682-092e-4266-9be4-51511c859637)

Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

* The LeNet-5 architecture is a classic convolutional neural network (CNN) designed by Yann LeCun et al. It is primarily known for its effectiveness in handwritten digit recognition tasks. This section provides an overview of the LeNet architecture and its components.
* The LeNet architecture consists of three convolutional layers (self.conv1, self.conv2, self.conv3) followed by two fully connected layers (self.fc1, self.fc2).
* LeNet uses tanh and sigmoid activation function.
* Input image size: 32*32*1

#### Convolutional Layer 1:
Input Channels: 1, Output Channels: 6, Stride: (1, 1), Kernel Size: (5, 5)

#### Convolutional Layer 2: 
Input Channels: 6, Output Channels: 16, Stride: (1, 1), Kernel Size: (5, 5)

#### Convolutional Layer 3:
Input Channels: 16, Output Channels: 120, Stride: (1, 1), Kernel Size: (5, 5)

#### Fully Connected Layer 1:
Input Features: 120, Output Features: 84

#### Fully Connected Layer 2 (Output Layer):
Input Features: 84, Output Features: Number of classes in the classification task (variable num_classes)

## VGG
The paper can be found at the following link: https://arxiv.org/pdf/1409.1556v6.pdf






