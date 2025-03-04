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


class CIFAR10Data:
    '''
    class for loading and splitting the cifar10 dataset
    for federated learning simulation.
    '''
    def __init__(
            self,
            iid : bool,
            validation_percent : float = 0.2,
            transform = default_transform
    ):
        
        self.iid = iid
        self.transform = transform
        
        self.load_data(validation_percent)


    def load_data(self, validation_percent):
        '''
        method for loading the dataset, and splitting into train, validation and test datasets
        '''
        dataset = CIFAR10(root=__DATA_DIR, train=True,
                   download=True, transform=self.transform)
        
        self.val_size = int(validation_percent * len(dataset))   # size of validation dataset
        self.train_size = len(dataset) - self.val_size                # size of training dataset

        #generate random training and validation sets
        trainset, valset = random_split(trainset, [self.train_size, self.val_size])

        testset = CIFAR10(root=__DATA_DIR, train=False, 
                               download=True, transform=self.transform)
        
        return trainset, valset, testset


    def split_data(self, num_clients):
        '''
        method for splitting the data depending on the FL setting
        '''
        if self.iid:
            client_indices = torch.tensor_split(torch.randperm(self.train_size), num_clients)
        else:
            raise NotImplementedError("Non-IID sampling has not been implemented yet")

        return client_indices