import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, chs=[32, 64, 128], num_lin=256):
        super(SmallCNN, self).__init__()
        num_lin = 128
        # Activation funtion
        self.relu = nn.ReLU()

        # Convcolutional layers
        self.ConvLayers = nn.ModuleList([
            nn.Conv2d(3, chs[0], kernel_size=3, padding=1),   # N x 3 x 32 x 32 -> N x 32 x 32 x 32
            nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # N x 32 x 16 x 16 -> N x 64 x 16 x 16
            nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1), # N x 64 x 8 x 8 -> N x 128 x 8 x 8
        ])

        # Linear layers
        self.fc1 = nn.Linear(chs[2] * 4 * 4, num_lin)      # first fully connected layer after convolutional layers
        self.fc2 = nn.Linear(num_lin, 10)               # final fully connected layer for output

        # batch normalization layers (for regularization)
        self.BatchNorms = nn.ModuleList([
            nn.BatchNorm2d(chs[0]),
            nn.BatchNorm2d(chs[1]),
            nn.BatchNorm2d(chs[2]),
        ])

        # regularization layers
        self.pool = nn.MaxPool2d(2, 2)      # max pooling layer for regularization
        self.dropout = nn.Dropout(0.2)      # dropout layer for regularization


    def forward(self, x):
        for conv, batchnorm in zip(self.ConvLayers, self.BatchNorms):
            # apply convolutional layer, then batch normalization, then ReLU, then max pooling
            x = batchnorm(conv(x))
            x = self.pool(self.relu(x))       # final shape: N x 128 x 4 x 4

        x = x.view(x.size(0), -1)       # reshape to N x 128*4*4
        x = self.relu(self.fc1(x))      # fully connected layer and ReLU
        x = self.dropout(x)             # apply dropout for regularization
        x = self.fc2(x)                 # final fully connected layer for output
        return x
