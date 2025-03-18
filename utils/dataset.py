import torch
import os
import random

from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
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
    print("loading data:")
    dataset = CIFAR10(root=__DATA_DIR, train=True,
                download=True, transform=transform)
    
    val_size = int(validation_percent * len(dataset))   # size of validation dataset
    train_size = len(dataset) - val_size                # size of training dataset

    #generate random training and validation sets
    trainset, valset = random_split(dataset, [train_size, val_size])

    testset = CIFAR10(root=__DATA_DIR, train=False, 
                            download=True, transform=transform)
    
    print(f"  number of training samples: {len(trainset)}")
    print(f"  number of validation samples: {len(valset)}")
    print(f"  number of test samples: {len(testset)}\n")

    return trainset, valset, testset



def split_data(data, num_clients=100, iid=True):
    '''
    method for splitting the data depending on the FL setting
    '''
    print("splitting data:")
    if iid:
        client_indices = torch.tensor_split(torch.randperm(len(data)), num_clients)
        print(f"  {num_clients} splits with {len(client_indices[0])} samples each\n")
    else:
        # raise NotImplementedError("Non-IID sampling has not been implemented yet")
        indices = data.indices
        labels = [data.dataset.targets[i] for i in indices]
        sorted_indices = [index for _, index in sorted(zip(labels, indices))]
        shards = [sorted_indices[i:i+len(sorted_indices)//num_clients//2] for i in range(0, len(sorted_indices), len(sorted_indices)//num_clients//2)]
        random.shuffle(shards)
        client_indices = tuple([shards[i] + shards[i+1] for i in range(0, len(shards), 2)])
        print(f" {num_clients} splits with {len(client_indices[0])} samples each\n")
    
    return client_indices