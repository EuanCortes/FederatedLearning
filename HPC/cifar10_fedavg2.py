import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch import optim
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import time
import copy

import os
import sys

############################################################
# ------------- Import from utils directory -------------- #
############################################################

current_dir = os.path.dirname(os.path.realpath(__file__))   # current directory
parent_dir = os.path.dirname(current_dir)                   # parent directory
model_dir = os.path.join(parent_dir, 'utils')               # model directory

if model_dir not in sys.path:
    sys.path.append(model_dir)  # add model to pythonpath

from model import SmallCNN
from dataset import load_data, split_data
from federated_learning import client_update, fed_avg

############################################################
# --------------------- End imports ---------------------- #
############################################################


def client_update(state_dict, device, dataloader, epochs, stream):
    '''
    client local update
    '''
    net = SmallCNN().to(device)
    net.load_state_dict(state_dict)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for Xtrain, Ytrain in dataloader:
            Xtrain, Ytrain = Xtrain.to(device, non_blocking=True), Ytrain.to(device, non_blocking=True)

            with torch.cuda.stream(stream):
                outputs = net(Xtrain)

                optimizer.zero_grad()
                loss = criterion(outputs, Ytrain)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    print("Client finished training", flush=True)
    cpu_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    return cpu_state_dict, epoch_loss / len(dataloader)
    

def gpu_process(cuda_id, client_ids, dataloaders, state_dict, epochs, return_dict, max_streams=4):
    '''
    gpu process for training
    '''
    device = torch.device(f"cuda:{cuda_id}")
    torch.cuda.set_device(device)
    print(f"gpu {cuda_id} training clients : {client_ids}", flush=True)
    streams = [torch.cuda.Stream() for _ in range(max_streams)]

    client_updates = {}
    for i in range(0, len(client_ids), max_streams):
        current_batch = client_ids[i: i + max_streams]
        # Launch local updates concurrently on available streams.
        for idx, client_id in enumerate(current_batch):
            stream = streams[idx]  # reuse stream from the pool
            client_updates[client_id] = client_update(state_dict, device, dataloaders[client_id], epochs, stream)

        torch.cuda.synchronize()

    return_dict[cuda_id] = client_updates
    print(f"GPU {cuda_id} finished training clients {client_ids}")



def validate(valloader, current_weights):
    '''
    validation of the model
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SmallCNN().to(device)
    net.load_state_dict(current_weights)
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total

    return val_acc


if __name__ == "__main__":

    ####################### hyperparameters ####################
    NUM_CLIENTS = 100
    C = 0.2
    CLIENTS_PER_ROUND = int(NUM_CLIENTS * C)
    MAX_ROUNDS = 200
    NUM_LOCAL_EPOCHS = 10
    #NUM_GPUS = torch.cuda.device_count()
    NUM_GPUS = 2
    ############################################################



    ####################### load data ##########################
    trainset, valset, testset = load_data(validation_percent=0.2)

    trainloader = DataLoader(trainset, batch_size=64,
                            shuffle=True, num_workers=0)

    valloader = DataLoader(valset, batch_size=64,
                            shuffle=True, num_workers=0)

    testloader = DataLoader(testset, batch_size=64,
                            shuffle=True, num_workers=0)
    ############################################################


    ############# prepare client partitions ####################
    client_indices = split_data(trainset, num_clients=NUM_CLIENTS, iid=True)

    clients = [
        DataLoader(Subset(trainset, client_indices[i]), 
                batch_size=64, 
                shuffle=True, 
                num_workers=4)
        for i in range(NUM_CLIENTS)
        ]
    ############################################################

    mp.set_start_method('spawn', force=True)

    print("Number of GPUs: ", NUM_GPUS)

    # store training loss and validation accuracy
    avg_train_loss = []
    val_accuracy = []

    round = 0
    val_acc = 0

    net = SmallCNN()
    current_weights = copy.deepcopy(net.state_dict())
    del net

    while val_acc < 0.77 and round < MAX_ROUNDS:

        round += 1

        #current_weights_cpu = {k: v.cpu() for k, v in current_weights.items()}

        client_ids = torch.randperm(NUM_CLIENTS)[:CLIENTS_PER_ROUND] # random selection of clients to participate
        local_weights = []
        temp_avg_loss = 0

        gpu_partitions = torch.chunk(client_ids, NUM_GPUS)

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        print("assigning clients to GPUs")
        for gpu_id, partition in enumerate(gpu_partitions):
            p = mp.Process(target=gpu_process, args=(gpu_id, partition, clients, current_weights, NUM_LOCAL_EPOCHS, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        client_updates = []
        for i in range(NUM_GPUS):
            client_updates.extend(return_dict[i].values())

        current_weights = fed_avg([state for state, loss in client_updates])

        if round % 1 == 0:
            val_acc = validate(valloader, current_weights)
            print(f"Round {round+1}, Validation accuracy: {val_acc}")

