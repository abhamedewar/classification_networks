import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from load_data import AIDDataset
from models.lenet import LeNet

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'.\AID_combine', type=str)
parser.add_argument('--csv_path', default=r'.\aid_dataset.csv', type=str)
args = parser.parse_args()

aid_dataset = AIDDataset(args.csv_path, args.data_path)

#checking if AIDDataset is working or not

# fig = plt.figure()

# for i, sample in enumerate(aid_dataset):
#     plt.imshow(sample[0])
#     plt.show()
#     if i == 3:
#         break



