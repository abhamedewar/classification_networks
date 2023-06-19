'''
This repository contains implementations of Convolutional Neural Network (CNN)
popular architectures from scratch.

Currently the repository supports the following CNN architectures:
1. LeNet
2. VGGNet
    VGG-11
    VGG-13
    VGG-16
    VGG-19
3. GoogLeNet  
4. AlexNet  

The codebase also provides support for conducting exploratory data analysis on the custom dataset.
The training module is integrated with tensorboard to visualize training loss, validation loss,
training accuracy, validation accuracy, intermediate training images, weights of layer in the model.

'''
import argparse
import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch import optim
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import CustomDataset
from utils import get_model
from eda import dataset_stats, plot_class_distribution, visualize_samples
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'.\AID_combine', type=str)
parser.add_argument('--csv_path', default=r'.\aid_dataset.csv', type=str)
parser.add_argument('--class_mapping', default=r'.\class_mapping.json', type=str)
parser.add_argument('--network_type', default='lenet', type=str, \
description="Currently supports: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet/inception','alexnet', 'lenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19'")

args = parser.parse_args()

with open(args.class_mapping, "r") as json_file:
    class_map = json.load(json_file)

data_df = pd.read_csv(args.csv_path)

# #EDA
# dataset_stats(data_df, class_map)
# plot_class_distribution(data_df, class_map)
# visualize_samples(data_df, args.data_path, class_map)

#setup device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(class_map)

#hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 1

model, transform = get_model(args.network_type, num_classes)
model.to(device)

custom_dataset = CustomDataset(args.csv_path, args.data_path, transform)
split = [int(len(custom_dataset)*0.9), int(len(custom_dataset)*0.1)]
train_set, valid_set = torch.utils.data.random_split(custom_dataset, split)
trainloader = DataLoader(train_set, batch_size, shuffle=True)
validloader = DataLoader(train_set, batch_size, shuffle=True)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#tensorboard summary writer
writer = SummaryWriter(f'runs/classification')

def train(model, trainloader, num_epoch):
    print("Start training...")
    i = 0
    for i in tqdm(range(num_epoch)):
        model.train()
        for batch, label in tqdm(trainloader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            writer.add_scalar("Loss/train", loss, i)
            loss.backward()
            optimizer.step()
            #visualize images 
            img_grid = torchvision.utils.make_grid(batch)
            writer.add_image('train/images', img_grid, i)
            #visualize weights of the convolution layer 3
            writer.add_histogram('conv 3', model.conv3.weight, i)
            curr_correct = (torch.argmax(pred, dim=1) == label).sum().item()
            curr_train_acc = float(curr_correct)/float(batch.shape[0])
            writer.add_scalar("Accuracy/train", curr_train_acc, i)
            i += 1

def evaluate(model, validloader):
    model.eval() 
    correct = 0
    i = 0
    with torch.no_grad():
        for batch, label in tqdm(validloader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = criterion(pred, label)
            writer.add_scalar("Loss/validation", loss, i)
            correct = (torch.argmax(pred, dim=1) == label).sum().item()
            valid_acc = correct/len(validloader.dataset)
            writer.add_scalar("Accuracy/validation", valid_acc, i)
            i += 1
    # acc = correct/len(validloader.dataset)
    # print("Evaluation accuracy: {}".format(acc))

train(model, trainloader, num_epochs)
evaluate(model, validloader)
