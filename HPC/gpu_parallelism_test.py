import torch
from torch import nn
#from torchvision.datasets import CIFAR10
#from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import time

import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)

parent_dir = os.path.dirname(current_dir)
print(parent_dir)

model_dir = os.path.join(parent_dir, 'model')
print(model_dir)

if model_dir not in sys.path:
    sys.path.append(model_dir)

from model import SmallCNN

