# Implementation of famous Convolutional Neural Network from Scratch

## Custom Dataset can be used to train the networks present in this repository. Just provide the dataset in the format given below and run any model of choice. 

## Dataset format:
#### Root directory
* The root directory should contain all the image files related to the dataset.
#### CSV File 
* The dataset is accompanied by a CSV file that contains two columns: "Image Name" and "Class". This CSV file serves as a reference to associate each image with its respective class or label.
* The "Image Name" column contains the names of the image files present in the root directory. Each entry in this column should uniquely identify an image file.
* The "Class" column represents the corresponding class or label for each image. It contains numerical values.

# Details of various CNN architectures

## GoogLeNet/Inception v1
The paper can be found at the following link: https://arxiv.org/pdf/1409.4842.pdf

* The main contribution of the GoogLeNet architecture is the inception module.
* Making the decision between pooling and convolutional operations, as well as determining the size and number of filters applied to the output of the previous layer, is a critical aspect of the network architecture design process.
* The Inception module addresses the trade-off between pooling and convolutional operations by running multiple operations simultaneously, such as pooling and convolution, and using multiple filter sizes (e.g., 3x3, 5x5). This approach allows for capturing diverse features without compromising performance.
* Image size- 224 x 224 x 3

### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/9db628f8-8f68-4957-bb03-274054879a68)

## VGG
The paper can be found at the following link: https://arxiv.org/pdf/1409.1556v6.pdf

* This repository consists of implementation of VGG-11, VGG-13, VGG-16 and VGG-19 architectures.
* Image size: 224 x 224 x 3
* Mean RGB value of training set is substracted from each image in training set.
* Batch size- 256
* L2- 5*10^-4
* Dropout ratio- 0.5
* Momentum- 0.9
* The learning rate was initially set to 10^−2 and then decreased by a factor of 10 when the validation set accuracy stopped improving.

### Architecture Details:

The different configurations of VGG are:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/bc8ff19b-a23e-4142-a461-a8479ff2d8dd)

* Kernel size for convolutional layers = 3 x 3, stride = 1
* Kernel size for MaxPooling is 2 x 2, stride = 2

## LeNet

The link to the paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

* The LeNet-5 architecture is a classic convolutional neural network (CNN) designed by Yann LeCun et al. It is primarily known for its effectiveness in handwritten digit recognition tasks. This section provides an overview of the LeNet architecture and its components.
* The LeNet architecture consists of three convolutional layers (self.conv1, self.conv2, self.conv3) followed by two fully connected layers (self.fc1, self.fc2).
* LeNet uses tanh and sigmoid activation function.
* Input image size: 32*32*1
  
### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/08e82682-092e-4266-9be4-51511c859637)

* Convolutional Layer 1: Input Channels: 1, Output Channels: 6, Stride: (1, 1), Kernel Size: (5, 5)
* Convolutional Layer 2: Input Channels: 6, Output Channels: 16, Stride: (1, 1), Kernel Size: (5, 5)
* Convolutional Layer 3: Input Channels: 16, Output Channels: 120, Stride: (1, 1), Kernel Size: (5, 5)
* Fully Connected Layer 1: Input Features: 120, Output Features: 84
* Fully Connected Layer 2 (Output Layer): Input Features: 84, Output Features: Number of classes in the classification task (variable num_classes)







