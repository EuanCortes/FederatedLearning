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

    NUM_CLIENTS = int(sys.argv[1])          # number of clients
    C = float(sys.argv[2])                  # fraction of clients to participate in each round
    CLIENTS_PER_ROUND = int(NUM_CLIENTS * C)    # number of clients to participate in each round
    MAX_ROUNDS = int(sys.argv[3])       # maximum number of rounds
    NUM_LOCAL_EPOCHS = int(sys.argv[4]) # number of local epochs
    

    # set up torch multiprocessing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()


    # load data and prepare dataloaders
    trainset, valset, testset = load_data(validation_percent=0.5)

    trainloader = DataLoader(trainset, batch_size=64,
                            shuffle=True, num_workers=4)
    
    valloader = DataLoader(valset, batch_size=64,
                            shuffle=True, num_workers=4)
    
    testloader = DataLoader(testset, batch_size=64,
                            shuffle=True, num_workers=4)

    # prepare client partitions
    client_indices = split_data(trainset, num_clients=NUM_CLIENTS, iid=True)

    clients = [
        DataLoader(Subset(trainset, client_indices[i]), batch_size=64, shuffle=True, num_workers=4)
        for i in range(NUM_CLIENTS)
    ]

    # store training loss and validation accuracy
    avg_train_loss = []
    val_accuracy = []

    round = 0
    val_acc = 0

    net = SmallCNN()
    current_weights = copy.deepcopy(net.state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    while val_acc < 0.77 and round < MAX_ROUNDS:

        round += 1

        #current_weights_cpu = {k: v.cpu() for k, v in current_weights.items()}

        client_ids = torch.randperm(NUM_CLIENTS)[:CLIENTS_PER_ROUND] # random selection of clients to participate
        local_weights = []
        temp_avg_loss = 0

        batched_clients = torch.split(client_ids, 2)

    
        for ids in batched_clients:
            processes = []
            for i in ids:
                cuda_id = i % torch.cuda.device_count()
                p = mp.Process(target=client_update, args=(cuda_id, i, clients[i], SmallCNN, current_weights, NUM_LOCAL_EPOCHS, return_dict))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            for i, (state_dict, loss) in return_dict.items():
                local_weights.append(state_dict)
                temp_avg_loss += loss
            
            return_dict.clear()


        avg_train_loss.append(temp_avg_loss / CLIENTS_PER_ROUND)
        
        print(f"Round {round} done")
        print(f"training loss: {avg_train_loss[-1]:.3f}")

        new_weights = fed_avg(local_weights)    
        print("Federated Averaging done")

        current_weights = new_weights


        # validation of model every 5 rounds
        #if round % 5 == 4:
        if True:
            val_acc = validate(valloader, current_weights)
            val_accuracy.append(val_acc)
            print(f"Validation accuracy: {val_acc:.3f}")

    print('Finished Training')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(np.arange(round), avg_train_loss, label="training loss")
    axs[0].set_title("Training Loss")
    axs[1].plot(np.arange(0, round, 5), np.array(val_accuracy) * 100, label="validation accuracy")
    axs[1].set_title("validation accuracy")

    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.savefig("training_plot.png")

    # test network


'''
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
'''