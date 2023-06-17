from torchvision import transforms
from models.lenet import LeNet
from models.vgg import VGG
from models.alexnet import AlexNet
from models.googlenet import InceptionV1
from models.resnet import resnet

vgg = {'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

def get_model(network_type, num_classes):
    if network_type == 'lenet':
        model = LeNet(num_classes=num_classes)
        transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])

    elif 'vgg' in network_type:
        in_channels = 3
        model = VGG(in_channels, num_classes, vgg[network_type])  
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) 

    elif network_type == 'alexnet':
        in_channels = 3
        model =  AlexNet(in_channels, num_classes)
        transform = transforms.Compose([
                transforms.Resize(227),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif network_type == 'googlenet' or network_type == 'inception':
        in_channels = 3
        model = InceptionV1(in_channels, num_classes)
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])  

    elif 'resnet' in network_type:
        in_channels = 3
        model = resnet(network_type, in_channels, num_classes)  
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])        
    
    return model, transform