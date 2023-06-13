from tkinter import font
import matplotlib.pyplot as plt
import os
from skimage import io


def dataset_stats(data_df, class_map):

    '''
    This function prints the dataset statistics- number of images, classes.

    Parameters:
        data_df: The dataframe with image name and class label.
        class_map: Mapping of class names and numbers.

    Returns: None
    '''

    distinct_count = data_df['Class'].nunique()
    total = len(data_df)

    print("Total number of images in the dataset: ", total)
    print("Total number of classes in the dataset: ", distinct_count)
    print("The different classes in the dataset are: ", list(class_map.values()))

def plot_class_distribution(data_df, class_map):

    '''
    This function plots and saves the class distribution of the dataset.

    Parameters:
        data_df: The dataframe with image name and class label.
        class_map: Mapping of class names and numbers.

    Returns: None
    '''    

    value_counts = data_df['Class'].value_counts().reset_index()
    categories = value_counts['Class']
    values = value_counts['count']
    categories = [class_map[str(categories[i])] for i in range(len(categories))]

    plt.figure(figsize=(12, 6))
    bar_width = 0.5  # Width of each bar
    plt.bar(categories, values, width=bar_width, align='edge', color="black")
    plt.xticks(rotation=80)
    plt.xlabel("Categories")
    plt.ylabel("# Images")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig('Class_distribution.png')
    plt.show()
    

def visualize_samples(data_df, root_dir, class_map):
    '''
    This function randomly plots samples.

    Parameters:
        data_df: The dataframe with image name and class label.
        root_dir: The folder with all the images from the dataset.
        class_map: Mapping of class names and numbers.

    Returns: None
    '''

    fig, ax = plt.subplots(2, 5, figsize=(20, 20))
    ax = ax.flatten()
    distinct_indices = data_df['Class'].drop_duplicates().index.tolist()
    for i, idx in enumerate(distinct_indices[:10]):
        img_name = os.path.join(root_dir, data_df.iloc[idx, 0])
        curr_image = io.imread(img_name)
        label = class_map[str(data_df.iloc[idx, 1])] 
        ax[i].imshow(curr_image)
        ax[i].set_title(label, fontsize=10, color='black', fontweight="bold", loc='center', pad=-40)
        ax[i].tick_params(axis='x', labelsize=6)
        ax[i].tick_params(axis='y', labelsize=6)
    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()
    plt.savefig('Samples.png')
    plt.show()
