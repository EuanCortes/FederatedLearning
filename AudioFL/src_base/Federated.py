import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
from .base_cnn import BaseCNN
import copy


class Client:
    """
    Class that represents clients in a federated learning system.
    The client should maintain a fixed local dataset, and
    contain a method that can be called to train the model locally
    when participating in a round of federated learning.
    """

    def __init__(
        self,
        net,
        data,
        batch_size,
        criterion,
        collate_fn=None,
        device=None,
        LR=0.001,
        weight_decay=0,
    ):
        """
        Constructor for the Client class.
        :param net:         the neural network model
        :param data:        the local dataset
        :param batch_size:  the batch size to use for training
        :param optimizer:   the optimizer to use for training
        :param collate_fn:  the function used for data resizing
        :param criterion:   the loss function to use for training
        :param device:      the device to run the training on (cpu or cuda)
        """
        self.net = net
        self.data = data
        self.batch_size = batch_size
        self.criterion = criterion
        if collate_fn == None:
            print("No collate function used, verify the behavior")
        self.collate_fn = collate_fn

        # set device to run the training on
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # create a DataLoader for the local dataset
        self.dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

        # set optimizer
        self.optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)

    def train(self, epochs, state_dict):
        """
        Training method that trains the model on the local dataset for a number of epochs.
        :param epochs:      the number of epochs to train
        :param state_dict:  the state dictionary of the global model

        :return:            the state dictionary of the trained model and the loss
        """

        self.net.to(self.device)
        self.net.load_state_dict(state_dict)  # load global weights
        self.net.train()  # set model to train mode

        for epoch in range(epochs):  # iterate thru local epochs

            epoch_loss = 0
            for Xtrain, Ytrain in self.dataloader:  # iterate thru local data
                Xtrain, Ytrain = Xtrain.to(self.device), Ytrain.to(self.device)

                outputs = self.net(Xtrain)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, Ytrain)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                del Xtrain, Ytrain, outputs, loss
                torch.cuda.empty_cache() 
        weights = copy.deepcopy(self.net.state_dict())
        self.net.to("cpu")
        return copy.deepcopy(self.net.state_dict()), epoch_loss / len(self.dataloader)

    def train_personnalized(self, epochs, global_shared_weights):
        """
        Args:
            num_epochs: local training iterations
            global_shared_weights: dict containing shared (Conv/BN) layers
        """

        self.net.load_state_dict(global_shared_weights, strict=False)
        self.net.to(self.device)
        self.net.train()

        epoch_loss = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        self.net.to("cpu")
        return self.net.state_dict(), epoch_loss / len(self.dataloader)


def client_update(
    cuda_id: int,
    dataloader: DataLoader,
    state_dict: dict,
    criterion=nn.CrossEntropyLoss(),
    epochs=5,
    lr=0.001,
    weight_decay=0,
):
    """
    client update method for parallelisation
    :param cuda_id:     the cuda device id to use
    :param dataloader:  the dataloader for the local dataset
    :param state_dict:  the state dictionary of the global model
    :param criterion:   the loss function to use for training
    :param epochs:      the number of epochs to train
    :param lr:          the learning rate for training
    :param weight_decay: the weight decay for training

    return:             the state dictionary of the trained model and the average batch loss
    """
    device = torch.device(f"cuda:{cuda_id}")
    net = BaseCNN().to_device()
    net.load_state_dict(state_dict)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

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

    return copy.deepcopy(net.state_dict()), epoch_loss / len(dataloader)


def FedAvg(state_dicts, weights=None):
    """
    Function that averages the weights of the models in the input list.
    :param state_dicts:     list of state dictionaries of the models to average
    :param weights:         list of weights to use for the averaging

    :return:                the state dictionary of the averaged model
    """
    avg_state = copy.deepcopy(state_dicts[0])  # copy the first model's weights

    for key in avg_state.keys():  # iterate thru the module weights

        if weights is not None:  # if weights are provided, use them for the averaging
            avg_state[key] = state_dicts[0][key] * weights[0]

        for i in range(1, len(state_dicts)):

            if (
                weights is not None
            ):  # if weights are provided, use them for the averaging
                avg_state[key] += state_dicts[i][key] * weights[i]

            else:
                avg_state[key] += state_dicts[i][key]

        avg_state[key] = avg_state[key] / len(state_dicts)

    return avg_state
