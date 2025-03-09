import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
import copy


def client_update(cuda_id : int,
                  client_id : int, 
                  dataloader : DataLoader, 
                  Net : nn.Module,
                  state_dict : dict, 
                  epochs : int,
                  return_dict : dict,
                  criterion=nn.CrossEntropyLoss(), 
                  lr=0.001, 
                  weight_decay=0,
                  debug=False):
    '''
    client update method for federated learning

    :param cuda_id:     the cuda device id to use
    :param dataloader:  the dataloader for the local dataset
    :param state_dict:  the state dictionary of the global model
    :param criterion:   the loss function to use for training
    :param epochs:      the number of epochs to train
    :param lr:          the learning rate for training
    :param weight_decay: the weight decay for training

    return:             the state dictionary of the trained model and the average batch loss
    '''
    device = torch.device(f"cuda:{cuda_id}")
    net = Net().to(device)
    net.load_state_dict(state_dict)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    if debug:
        print(f"Client {client_id} training on device {device}")

    for epoch in range(epochs):
        epoch_loss = 0
        for Xtrain, Ytrain in dataloader:
            Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)

            outputs = net(Xtrain)

            optimizer.zero_grad()
            loss = criterion(outputs, Ytrain)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    
    return_dict[client_id] = (state_dict, epoch_loss / len(dataloader))



def fed_avg(state_dicts, weights=None):
    '''
    Function that averages the weights of the models in the input list.

    :param state_dicts:     list of state dictionaries of the models to average
    :param weights:         list of weights to use for the averaging

    :return:                the state dictionary of the averaged model
    '''
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
