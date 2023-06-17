# Implementation of State-of-the-Art Convolutional Neural Network from Scratch

## GoogLeNet, ResNet, AlexNet, LeNet, VGG11, VGG13, VGG16, VGG19
## Custom Dataset can be used to train the networks present in this repository. Just provide the dataset in the format given below and run any model of choice. 

## Dataset format:
#### Root directory
* The root directory should contain all the image files related to the dataset.
#### CSV File 
* The dataset is accompanied by a CSV file that contains two columns: "Image Name" and "Class". This CSV file serves as a reference to associate each image with its respective class or label.
* The "Image Name" column contains the names of the image files present in the root directory. Each entry in this column should uniquely identify an image file.
* The "Class" column represents the corresponding class or label for each image. It contains numerical values.
#### Class Mapping JSON:
* Json file with class mapping. Refer class_mapping.json.
  
## Running the code:

Currently the code supports the following CNN architectures: ['googlenet', 'resnet', 'alexnet', 'lenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19'].

Once you have the dataset ready in the above format just run the code by executing the following command:

Install the dependencies:

```
pip install -r requirements.txt
```

Run the code:

```
python main.py --data_path <folder with all images> --csv_path <path to csv file> --class_mapping <json file with class mapping> --network_type <cnn type>
```

# Details of various CNN architectures

## GoogLeNet/Inception v1
The paper can be found at the following link: https://arxiv.org/pdf/1409.4842.pdf

* The main contribution of the GoogLeNet architecture is the inception module.
* Making the decision between pooling and convolutional operations, as well as determining the size and number of filters applied to the output of the previous layer, is a critical aspect of the network architecture design process.
* The Inception module addresses the trade-off between pooling and convolutional operations by running multiple operations simultaneously, such as pooling and convolution, and using multiple filter sizes (e.g., 3x3, 5x5). This approach allows for capturing diverse features without compromising performance.
* **Input image size- 224 x 224 x 3**

### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/5e704450-dff8-4d13-9336-e4eaea6402b4)

* **#3×3 reduce and #5×5 reduce** stands for the number of 1×1 filters in the reduction layer used before the 3×3 and 5×5 convolutions.
* **pool proj column** is the number of 1×1 filtersafter the built-in max-pooling.

### Inception block

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/165d811f-e440-4b8d-ba38-36c222b3fac4)

## ResNet

The paper can be found at the following link: https://arxiv.org/abs/1512.03385

* The problem with deeper networks is that it can cause vanishing gradient problem.
* The main idea behind the paper is to use skip connections to address the problem of vanishing gradient.(Introduced residual blocks as shown below)
* The way to create ResNet is taking multiple residual blocks and stacking them to create a deep neural network.
* **Input image size- 224 x 224 x 3**

### Residual Block

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/4c43f199-28de-497e-89bf-3ea070aab11e)

### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/05ab5524-6577-4164-82f1-bf55ac05035d)

## AlexNet

* **Input image size: 32*32*1**
* 5 convolutional layers, 3 fully connected layers

### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/70955335-6867-4cbf-9ff9-5f5c45255e83)

Source: https://www.mdpi.com/2072-4292/9/8/848

## VGG
The paper can be found at the following link: https://arxiv.org/pdf/1409.1556v6.pdf

* This repository consists of implementation of VGG-11, VGG-13, VGG-16 and VGG-19 architectures.
* **Input image size: 224 x 224 x 3**
* Mean RGB value of training set is substracted from each image in training set.
* Batch size- 256
* L2- 5*10^-4
* Dropout ratio- 0.5
* Momentum- 0.9
* The learning rate was initially set to 10^−2 and then decreased by a factor of 10 when the validation set accuracy stopped improving.

### Architecture Details:

The different configurations of VGG are:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/bc3a8f97-d94b-4831-b41f-a355b0a5c2c2)

* Kernel size for convolutional layers = 3 x 3, stride = 1
* Kernel size for MaxPooling is 2 x 2, stride = 2

## LeNet

The link to the paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

* The LeNet-5 architecture is a classic convolutional neural network (CNN) designed by Yann LeCun et al. It is primarily known for its effectiveness in handwritten digit recognition tasks. This section provides an overview of the LeNet architecture and its components.
* The LeNet architecture consists of three convolutional layers (self.conv1, self.conv2, self.conv3) followed by two fully connected layers (self.fc1, self.fc2).
* LeNet uses tanh and sigmoid activation function.
* **Input image size: 32*32*1**
  
### Architecture Details:

![image](https://github.com/abhamedewar/classification_networks/assets/20626950/2cf2b001-a663-450b-b246-b4e56929296b)

* Convolutional Layer 1: Input Channels: 1, Output Channels: 6, Stride: (1, 1), Kernel Size: (5, 5)
* Convolutional Layer 2: Input Channels: 6, Output Channels: 16, Stride: (1, 1), Kernel Size: (5, 5)
* Convolutional Layer 3: Input Channels: 16, Output Channels: 120, Stride: (1, 1), Kernel Size: (5, 5)
* Fully Connected Layer 1: Input Features: 120, Output Features: 84
* Fully Connected Layer 2 (Output Layer): Input Features: 84, Output Features: Number of classes in the classification task (variable num_classes)







