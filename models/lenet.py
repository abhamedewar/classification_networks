from torch import nn
import torch

class LeNet(nn.Module):
    '''
    LeNet-5 is a classic convolutional neural network architecture
    designed by Yann LeCun et al. for handwritten digit recognition.

    Args:
        num_classes: The number of output classes.
    '''
    def __init__(self, num_classes):

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels= 120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2=nn.Linear(in_features=84, out_features=num_classes)

        self.pooling_layer = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        #Convolution Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        #Convolution Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_layer(x)
        x = self.dropout(x)
        #Convolution Layer 3
        x = self.conv3(x)
        x = self.relu(x)
        #flatten x
        x = x.reshape(x.shape[0], -1)
        #Fully connected layer 1
        x = self.fc1(x)
        x = self.relu(x)
        #Fully connected layer 2
        x = self.fc2(x)

        return x

##testing LeNet
x = torch.randn(64, 1, 32, 32)
model = LeNet(30)
# print(model(x).shape) ----> should give 64*30 