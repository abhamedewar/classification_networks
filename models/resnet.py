from torch import nn
import torch

class ResidualBlock1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        identity = x
        val = self.conv1(x)
        val = self.batch_norm1(val)
        val = self.relu(val)

        val = self.conv2(val)
        val = self.batch_norm2(val)

        val += identity
        val = self.relu(val)

        return val

class ResidualBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        
        identity = x
        val = self.conv1(x)
        val = self.batch_norm1(val)
        val = self.relu(val)

        val = self.conv2(val)
        val = self.batch_norm2(val)
        val = self.relu(val)

        val = self.conv3(val)
        val = self.batch_norm3(val)

        val += identity
        val = self.relu(val)

        return val

class ResNet(nn.Module):

    def __init__(self, layers, in_channels, num_classes, res_block):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv = self.architecture(layers, in_channels, num_classes, res_block)
        pass

    def forward(self):
        pass

    def architecture(layers, in_channels, num_classes, res_block):
        pass


def resnet(arch_type, in_channels, num_classes):

    if arch_type == 'resnet18':
        layers = [2, 2, 2, 2]
        res_block = ResidualBlock1

    elif arch_type == 'resnet34':
        layers = [3, 4, 6, 3]
        res_block = ResidualBlock1

    elif arch_type == 'resnet50':
        layers = [3, 4, 6, 3]
        res_block = ResidualBlock2

    elif arch_type == 'resnet101':
        layers = [3, 4, 23, 3]
        res_block = ResidualBlock2

    elif arch_type == 'resnet152':
        layers = [3, 8, 36, 3]
        res_block = ResidualBlock2
    
    model = ResNet(layers, in_channels, num_classes, res_block)
    return model



    
