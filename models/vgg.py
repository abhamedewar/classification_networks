from torch import nn

class VGG(nn.Module):
    '''
    This class implements the VGGNet architecture for image classification.
    VGGNet is known for its deep architecture with 11, 13, 16 or 19 weight 
    layers and the use of small 3x3 convolutional filters.
    Args:
        num_classes: The number of output classes.
    '''
    def __init__(self, in_channels, num_classes, vgg_layers):

        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.vgg_layer_info = vgg_layers
        self.conv_layers = self.architecture()
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def architecture(self):
        layers = []
        in_channels = self.in_channels
        for x in self.vgg_layer_info:
            if type(x) == int:
                out_channels = x
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
                in_channels = x
            if type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

# test vgg
# x = torch.randn(64, 3, 224, 224)
# model = VGG(3, 30, [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
# print(model)
# print(model(x).shape) ----------> should give 64*30
