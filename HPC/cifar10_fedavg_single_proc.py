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
from federated_learning import federated_sim

############################################################
# --------------------- End imports ---------------------- #
############################################################


if __name__ == "__main__":

    ####################### hyperparameters ####################
    num_clients = [100, 500]
    Cs = [[0.1, 0.2], [0.05, 0.1]]

    max_rounds = [200, 500]
    num_local_epochs = [5, 5]

    lrs = [0.001, 0.001]

    experiment_name = "smaller_network"

    os.mkdir(f"outputs/figures/{experiment_name}")
    ############################################################


    for NUM_CLIENTS, Clist, MAX_ROUNDS, NUM_LOCAL_EPOCHS, lr in zip(num_clients, Cs, max_rounds, num_local_epochs, lrs):
        for C in Clist:
            final_round, fig1, fig2 = federated_sim(num_clients=NUM_CLIENTS,
                                            frac_participants=C,
                                            max_rounds=MAX_ROUNDS,
                                            num_local_epochs=NUM_LOCAL_EPOCHS,
                                            lr=lr,
                                            )

            fig1.savefig(f"outputs/figures/{experiment_name}/err_acc_N_{NUM_CLIENTS:d}__C_{C:.2f}__iters_{final_round:d}.png", dpi=200)
            fig2.savefig(f"outputs/figures/{experiment_name}/hist_N_{NUM_CLIENTS:d}__C_{C:.2f}__iters_{final_round:d}.png", dpi=200)

    print("Finished experiment.")