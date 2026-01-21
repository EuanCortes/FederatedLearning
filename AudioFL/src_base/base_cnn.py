### File containing the base implementation of the CNN class ###

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


class BaseCNN(nn.Module):
    """Base class to construct CNN for audio FL
    Input should be of size 1x2**15
    """

    def __init__(self, input_size=2**15, nb_class=10):
        super(BaseCNN, self).__init__()
        self.input_size = input_size
        self.nb_class = nb_class
        self.training_epochs = 30
        self.ConvLayers = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.Conv1d(16, 64, kernel_size=3, padding=1),
                nn.Conv1d(64, 256, kernel_size=3, padding=1),
            ]
        )
        self.BatchNorms = nn.ModuleList(
            [
                nn.BatchNorm1d(16),
                nn.BatchNorm1d(64),
                nn.BatchNorm1d(256),
            ]
        )

        flattened_size = 256 * (input_size // 8)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, self.nb_class)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

    def forward(self, x):
        for conv, batch in zip(self.ConvLayers, self.BatchNorms):  # ← Add zip()
            x = batch(conv(x))
            x = self.pool(self.relu(x))  # Also fix: self.relu() not self.relu
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def to_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        net = self.to(device)
        return net

    def train_mod(
        self, nb_epoch=30, trainLoader=DataLoader([]), valLoader=DataLoader([])
    ):
        net = self.to_device()
        self.training_epochs = nb_epoch
        model_weights = []
        print(f"Starting training over {self.training_epochs}")

        # Training parameters #
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        batch_size = 64  ## Check if correct
        training_loss = []
        val_accuracy = []

        for epoch in range(self.training_epochs):

            net.train()
            running_loss = 0.0
            for i, (input, labels) in enumerate(trainLoader):
                # -- get the inputs; data is a list of [inputs, labels]
                inputs, labels = input.to(self.device), labels.to(self.device)

                # -- zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # -- print statistics
                running_loss += loss.item()

            # writer.add_scalar("Batch_Loss/train", running_loss / len(trainloader), epoch)   # TENSORBOARD
            training_loss.append(running_loss / len(trainLoader))

            if epoch % 1 == 0:
                net.eval()

                correct = 0
                total = 0

                with torch.no_grad():
                    for images, labels in valLoader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        # -- calculate outputs by running images through the network
                        outputs = net(images)
                        # -- the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                epoch_accuracy = 100 * correct // total
                val_accuracy.append(epoch_accuracy)
                print(
                    f"Epoch {epoch+1}/{nb_epoch},\n  Train loss: {training_loss[epoch]},\n  Validation Accuracy: {epoch_accuracy}%"
                )

        return (training_loss, val_accuracy)


### Usage ###
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")
# net = SmallCNN().to(device)
