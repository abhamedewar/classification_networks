import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from load_data import CustomDataset
from models.lenet import LeNet
from eda import dataset_stats, plot_class_distribution, visualize_samples
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'.\AID_combine', type=str)
parser.add_argument('--csv_path', default=r'.\aid_dataset.csv', type=str)
parser.add_argument('--class_mapping', default=r'.\class_mapping.json', type=str)
parser.add_argument('--network_type', default='lenet', type=str)
args = parser.parse_args()

with open(args.class_mapping, "r") as json_file:
    class_map = json.load(json_file)

data_df = pd.read_csv(args.csv_path)

custom_dataset = CustomDataset(args.csv_path, args.data_path)

#checking if CustomDataset is working or not

# fig = plt.figure()
# for i, sample in enumerate(custom_dataset):
#     plt.imshow(sample[0])
#     plt.show()
#     if i == 3:
#         break

#EDA

# dataset_stats(data_df, class_map)
# plot_class_distribution(data_df, class_map)
# visualize_samples(data_df, args.data_path, class_map)

