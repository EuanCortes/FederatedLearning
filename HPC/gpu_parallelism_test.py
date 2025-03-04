import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import time

import os
import sys

############################################################
# ------------- Import from model directory -------------- #
############################################################

current_dir = os.path.dirname(os.path.realpath(__file__))   # current directory
parent_dir = os.path.dirname(current_dir)                   # parent directory
model_dir = os.path.join(parent_dir, 'model')               # model directory

if model_dir not in sys.path:
    sys.path.append(model_dir)  # add model to pythonpath

from model import SmallCNN

############################################################
# --------------------- End imports ---------------------- #
############################################################

data_path = os.path.join(parent_dir, "data")



net = SmallCNN()

print(net)
