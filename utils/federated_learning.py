import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch import nn
import copy
from dataset import load_data, split_data
from model import SmallCNN

import matplotlib.pyplot as plt
import seaborn as sns


def client_update(device : torch.device,
                  dataloader : DataLoader, 
                  Net : nn.Module,
                  state_dict : dict,
                  epochs : int,
                  criterion=nn.CrossEntropyLoss(), 
                  lr=0.001, 
                  weight_decay=0):
    '''
    client update method for federated learning

    :param device:      the device to train the model on
    :param dataloader:  the dataloader for the local dataset
    :param net:         the neural network model
    :param epochs:      the number of epochs to train
    :param criterion:   the loss function to use for training
    :param lr:          the learning rate for training
    :param weight_decay: the weight decay for training

    return:             the state dictionary of the trained model and the average batch loss
    '''

    net = Net().to(device)  # initialise neural net and send to device
    net.load_state_dict(state_dict) # load current state dict
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # init optimizer
    net.train() # set net to train

    # iterate thru number of local epochs
    for epoch in range(epochs):
        epoch_loss = 0

        # iterate thru local dataset
        for Xtrain, Ytrain in dataloader:
            Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)

            outputs = net(Xtrain)

            optimizer.zero_grad()
            loss = criterion(outputs, Ytrain)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # move state dict to cpu
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    
    return state_dict, epoch_loss / len(dataloader)



def fed_avg(state_dicts, weights=None):
    '''
    Function that averages the weights of the models in the input list.

    :param state_dicts:     list of state dictionaries of the models to average
    :param weights:         list of weights to use for the averaging

    :return:                the state dictionary of the averaged model
    '''

    if weights is not None:
        assert torch.sum(weights) == 1, "weights should sum to 1!"
    avg_state = copy.deepcopy(state_dicts[0])   # copy the first model's weights 

    for key in avg_state.keys():    # iterate thru the module weights

        if weights is not None:     # if weights are provided, use them for the averaging
            avg_state[key] = state_dicts[0][key] * weights[0]

        for i in range(1, len(state_dicts)):
            
            if weights is not None:     # if weights are provided, use them for the averaging
                avg_state[key] += state_dicts[i][key] * weights[i]

            else:
                avg_state[key] += state_dicts[i][key]

        avg_state[key] = avg_state[key] / len(state_dicts)

    return avg_state


def validate(net, current_weights, valloader):
    '''
    validation of the model
    '''

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net = Net().to(device)
    device = next(net.parameters()).device
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


def federated_sim(num_clients : int,
                  frac_participants : float,
                  max_rounds : int,
                  num_local_epochs : int,
                  lr=0.001,
                  ):
    '''
    function to simulate federated learning on single process
    '''

    clients_per_round = int(num_clients * frac_participants)    # number of clients participating in each round

    print(f"Number of clients: {num_clients}")
    print(f"Number of clients per round: {clients_per_round}")

    ####################### load data ##########################
    trainset, valset, testset = load_data(validation_percent=0.2) # load datasets

    valloader = DataLoader(valset, batch_size=64,           # validation set dataloader
                            shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=64,         # test set dataloader
                            shuffle=True, num_workers=0)
    ############################################################


    ############# prepare client partitions ####################
    client_indices = split_data(trainset, num_clients=num_clients, iid=True)

    clientloaders = [
        DataLoader(Subset(trainset, client_indices[i]), 
                batch_size=64, 
                shuffle=True, 
                num_workers=0)
        for i in range(num_clients)
        ]
    ############################################################


    ################### set up simulation ######################
    # store training loss and validation accuracy
    avg_train_loss = []
    val_accuracy = []
    
    client_counter = np.zeros(num_clients, dtype=int)

    current_round = 0
    val_acc = 0

    net = SmallCNN()
    current_weights = copy.deepcopy(net.state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    print("network architecture:")
    print(net)

    print_every = max_rounds // 20
    ############################################################


    while val_acc < 0.77 and current_round < max_rounds:

        current_round += 1

        client_ids = torch.randperm(num_clients)[:clients_per_round].tolist() # random selection of clients to participate
        local_weights = []
        temp_avg_loss = 0

        for client in client_ids:
            client_counter[client] += 1
            weights, loss = client_update(device, clientloaders[client], SmallCNN, current_weights, num_local_epochs, lr=lr)
            local_weights.append(weights)
            temp_avg_loss += loss
        
        avg_train_loss.append(temp_avg_loss / clients_per_round)

        current_weights = fed_avg(local_weights)    

        val_acc = validate(net, current_weights, valloader)
        val_accuracy.append(val_acc)

        if current_round % print_every == print_every - 1:
            print(f"Round {current_round} done")
            print(f"training loss: {avg_train_loss[-1]:.3f}")
            print(f"Validation accuracy: {val_acc:.3f}")

    print(f'Finished Training in {current_round} rounds')
    print(f"training loss: {avg_train_loss[-1]:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")


    fig1, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(np.arange(current_round), avg_train_loss, label="training loss")
    axs[0].set_title("Training Loss")
    axs[1].plot(np.arange(current_round), np.array(val_accuracy) * 100, label="validation accuracy")
    axs[1].set_title("validation accuracy")

    client_counter = np.sort(client_counter)
    fig2, hist_ax = plt.subplots(figsize=(10,6))
    sns.histplot(client_counter.astype(str), ax=hist_ax)
    hist_ax.set_xlabel('Number of Selections')
    hist_ax.set_ylabel('Number of Clients')
    hist_ax.set_title('Histogram of Client Selections')

    return current_round, fig1, fig2
