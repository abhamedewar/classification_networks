import argparse
import imp
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
from models.lenet import LeNet
from eda import dataset_stats, plot_class_distribution, visualize_samples


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'.\AID_combine', type=str)
parser.add_argument('--csv_path', default=r'.\aid_dataset.csv', type=str)
parser.add_argument('--class_mapping', default=r'.\class_mapping.json', type=str)
parser.add_argument('--network_type', default='lenet', type=str)
args = parser.parse_args()

with open(args.class_mapping, "r") as json_file:
    class_map = json.load(json_file)

data_df = pd.read_csv(args.csv_path)

# custom_dataset = CustomDataset(args.csv_path, args.data_path)
# #checking if CustomDataset is working or not

# fig = plt.figure()
# for i, sample in enumerate(custom_dataset):
#     plt.imshow(sample[0])
#     plt.show()
#     if i == 3:
#         break

# exit()

#EDA

# dataset_stats(data_df, class_map)
# plot_class_distribution(data_df, class_map)
# visualize_samples(data_df, args.data_path, class_map)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(class_map)

#hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 1

if args.network_type == 'lenet':
    model = LeNet(num_classes=num_classes)
    model.to(device)
    resize = transforms.Resize(32)

#if image coloured
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#if image is Grayscale
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
transform = transforms.Compose([
                transforms.Grayscale(),
				resize,
				transforms.ToTensor(),
				normalize,
			])

custom_dataset = CustomDataset(args.csv_path, args.data_path, transform)
split = [int(len(custom_dataset)*0.9), int(len(custom_dataset)*0.1)]
train_set, valid_set = torch.utils.data.random_split(custom_dataset, split)
trainloader = DataLoader(train_set, batch_size, shuffle=True)
validloader = DataLoader(train_set, batch_size, shuffle=True)


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(model, trainloader, num_epoch):
    print("Start training...")
    for i in tqdm(range(num_epoch)):
        model.train()
        for batch, label in tqdm(trainloader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

def evaluate(model, validloader):
    model.eval() 
    correct = 0
    with torch.no_grad():
        for batch, label in tqdm(validloader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(validloader.dataset)
    print("Evaluation accuracy: {}".format(acc))

train(model, trainloader, num_epochs)
evaluate(model, validloader)
