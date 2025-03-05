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


if __name__ == "__main__":

    # load data and prepare dataloaders
    trainset, valset, testset = load_data()

    trainloader = DataLoader(trainset, batch_size=64,
                            shuffle=True, num_workers=4)
    
    valloader = DataLoader(valset, batch_size=64,
                            shuffle=True, num_workers=4)
    
    testloader = DataLoader(testset, batch_size=64,
                            shuffle=True, num_workers=4)

    # select cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    #initialise net, optimizer and criterion
    net = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)



    training_loss = []
    val_accuracy = []

    epoch = 0
    valid_acc = 0

    while valid_acc < 0.77:

        epoch += 1

        net.train()
        running_loss = 0.0
        for i, (input, labels) in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = input.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
        training_loss.append(running_loss / len(trainloader))

        #if epoch % 2 == 1:
        if True:
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
            val_accuracy.append(val_acc)
            print(f"Epoch {epoch},\n  Train loss: {training_loss[-1]},\n  Validation Accuracy: {val_acc * 100:.1f}%")

    print('Finished Training')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(np.arange(epoch), training_loss, label="training loss")
    axs[1].plot(np.arange(epoch), np.array(val_accuracy) / 100, label="validation accuracy")

    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.show()

    # test network

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
