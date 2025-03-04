import torch
import os

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms

current_dir = os.path.dirname(os.path.realpath(__file__))   # current directory
parent_dir = os.path.dirname(current_dir)                   # parent directory
__DATA_DIR = os.path.join(parent_dir, "data")

# define the transformation of the data. 
default_transform = transforms.Compose(
    [transforms.ToTensor(),     # convert the image to a pytorch tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalise the image with mean and std of 0.5


def load_data(validation_percent=0.2, transform=default_transform):
    '''
    function for loading the dataset, and splitting into train, validation and test datasets
    '''
    dataset = CIFAR10(root=__DATA_DIR, train=True,
                download=True, transform=transform)
    
    val_size = int(validation_percent * len(dataset))   # size of validation dataset
    train_size = len(dataset) - val_size                # size of training dataset

    #generate random training and validation sets
    trainset, valset = random_split(trainset, [train_size, val_size])

    testset = CIFAR10(root=__DATA_DIR, train=False, 
                            download=True, transform=transform)
    
    return trainset, valset, testset



def split_data(data, num_clients=100, iid=True):
    '''
    method for splitting the data depending on the FL setting
    '''
    if iid:
        client_indices = torch.tensor_split(torch.randperm(len(data)), num_clients)
    else:
        raise NotImplementedError("Non-IID sampling has not been implemented yet")

    return client_indices