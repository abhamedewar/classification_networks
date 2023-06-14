from torch import nn 

class GoogLeNet(nn.Module):

    '''
    GoogLeNet is also called as Inception v1
    '''
    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes

    def forward(x):
        pass


