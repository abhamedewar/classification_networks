from logging import root
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

class AIDDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.aid_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.aid_frame)
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.root_dir, self.aid_frame.iloc[index, 0])
        image = io.imread(img_name)
        label = torch.tensor(self.aid_frame.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return [image, label]


