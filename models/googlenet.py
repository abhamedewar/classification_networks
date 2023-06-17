from functools import reduce
from turtle import forward
from torch import nn 
import torch

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_1, reduce_3, out_3, reduce_5, out_5, out_pool_proj):
        super(InceptionBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, out_1, kernel=1, stride=1)
        self.block2 = nn.Sequential(ConvBlock(in_channels, reduce_3, kernel=1, stride=1),
                                    ConvBlock(reduce_3, out_3, kernel=3, stride=1, padding=1))
        self.block3 = nn.Sequential(ConvBlock(in_channels, reduce_5, kernel=1, stride=1),
                                    ConvBlock(reduce_5, out_5, kernel=5, stride=1, padding=2))
        self.block4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    ConvBlock(in_channels, out_pool_proj, kernel=1, stride=1))


    def forward(self, x):
        #1 because we want to concatenate all the filters
        return torch.cat([self.block1(x), self.block2(x), self.block3(x), self.block4(x)], 1)

class AuxilaryClassifier(nn.Module):

    def __init__(self, in_channel, num_classes):
        super(AuxilaryClassifier, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=5, stride=3) #different
        self.conv = ConvBlock(in_channel, 128, kernel=1, stride=1)
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        #2048 because the x will be 4*4*128
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):

        x = self.pooling(x)
        x = self.relu(self.conv(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class InceptionV1(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(InceptionV1, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, kernel=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel=1, stride=1)
        self.conv3 = ConvBlock(64, 192, kernel=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        # aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        #we can also return the two other auxilary outputs
        return x

#test inception V1
# x = torch.randn(64, 3, 224, 224)
# model = InceptionV1(3, 30)
# print(model(x).shape)