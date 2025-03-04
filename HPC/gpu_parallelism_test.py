import torch
from torch import nn
#from torchvision.datasets import CIFAR10
#from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
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
from dataset import load_data, split_data

############################################################
# --------------------- End imports ---------------------- #
############################################################


def client_update(id, dataloader, epochs=10, lr=1e-3, criterion=nn.CrossEntropyLoss()):

    device = torch.device(f"cuda:{id}")
    net = SmallCNN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):     # iterate thru local epochs
            
            epoch_loss = 0
            for Xtrain, Ytrain in dataloader:     # iterate thru local data
                Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)

                outputs = net(Xtrain)

                optimizer.zero_grad()
                loss = criterion(outputs, Ytrain)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() 


if __name__ == "__main__":

    num_clients = 100
    trainset, valset, testset = load_data(validation_percent=0.2)
    client_indices = split_data(trainset, num_clients=num_clients, iid=True)
     
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    print(f"num cuda devices: {torch.cuda.device_count()}")

    for i in range(10):
        print(f"training {i+1} clients on 1 gpu")
        t1 = time.perf_counter()
        dataloaders = [
             DataLoader(Subset(trainset, client_indices[j]), batch_size=32, shuffle=True) for j in range(i+1)
             ]
        
        for j, dataloader in enumerate(dataloaders):
            cuda_id = j % torch.cuda.device_count()
            p = mp.Process(target=client_update, args=(cuda_id, dataloader,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        t2 = time.perf_counter()
        print(f"Time taken to train {len(dataloaders)} client(s) on 1 gpu: {t2-t1:.3f} seconds\n")