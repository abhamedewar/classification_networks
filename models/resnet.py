from torch import nn
import torch

class ResidualBlock1(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.skip_connection = None

        #if the input size changes or the number of channels changes
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * out_channels)
                ) 

    def forward(self, x):
        
        identity = x
        val = self.conv1(x)
        val = self.batch_norm1(val)
        val = self.relu(val)

        val = self.conv2(val)
        val = self.batch_norm2(val)

        if self.skip_connection:
            identity = self.skip_connection(identity)

        val += identity
        val = self.relu(val)

        return val

class ResidualBlock2(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.skip_connection = None
        #if the input size changes or the number of channels changes
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * out_channels)
                ) 
        

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

        if self.skip_connection:
            identity = self.skip_connection(identity)

        val += identity
        val = self.relu(val)

        return val

class ResNet(nn.Module):

    def __init__(self, layers, in_channels, num_classes, res_block):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self.form_layer(layers[0], out_channels = 64, res_block = res_block, stride = 1)
        self.layer2 = self.form_layer(layers[1], out_channels = 128, res_block = res_block, stride = 2)
        self.layer3 = self.form_layer(layers[2], out_channels = 256, res_block = res_block, stride = 2)
        self.layer4 = self.form_layer(layers[3], out_channels = 512, res_block = res_block, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * res_block.expansion, num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        print(x.shape)

        x = self.layer1(x)

        print(x.shape)

        x = self.layer2(x)

        print(x.shape)

        x = self.layer3(x)

        print(x.shape)

        x = self.layer4(x)

        print(x.shape)

        x = self.avgpool(x)

        print(x.shape)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def form_layer(self, repeat, out_channels, res_block, stride):

        layers = []
        layers.append(res_block(self.in_channels, out_channels, stride))
        #the in_channels change her
        self.in_channels = res_block.expansion * out_channels
        for _ in range(repeat - 1):
            layers.append(res_block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)


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


#testing resnet
# x = torch.randn(1, 3, 224, 224)
# model = resnet("resnet18", 3, 30)
# print(model(x).shape)

    
