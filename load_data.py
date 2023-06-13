import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

class CustomDataset(Dataset):
    '''
    A custom PyTorch dataset for loading image and label data.

    Args:
        csv_file: The path to the CSV file containing image and label information.
        root_dir: The root directory of the image files.
        transform: A function/transform to be applied on the image data.

    Attributes:
        data_df: The DataFrame containing the image and label data.
        root_dir: The root directory of the image files.
        transform: A function/transform to be applied on the image data.

    '''

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Initializes the CustomDataset by reading image and label data from a CSV file.

        Args:
            csv_file: The path to the CSV file containing image and label information.
            root_dir: The root directory of the image files.
            transform: A function/transform to be applied on the image data.
        '''
        
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        '''
        Returns the length of the dataset.
        Returns:
            int: The number of samples in the dataset.
        '''

        return len(self.data_df)
    
    def __getitem__(self, index):
        '''
        Retrieves a sample from the dataset at the given index.
        Args:
            index: The index of the sample to retrieve.

        Returns:
            list: A list containing the image data and label for the sample.
        '''

        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.root_dir, self.data_df.iloc[index, 0])
        image = io.imread(img_name)
        label = torch.tensor(self.data_df.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return [image, label]


