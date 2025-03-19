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

    # num_clients = [1, 10, 100, 500]
    num_clients = [500]
    # Cs = [[1.], [0.5, 1.], [0.1, 0.2], [0.05, 0.1]]
    Cs =  [[1.]]

    # max_rounds = [10, 20, 100, 300]
    max_rounds = [10]
    # num_local_epochs = [5, 5, 5, 5]
    num_local_epochs = [5]


    # lrs = [0.001, 0.001, 0.002, 0.003]
    lrs = [0.001]
    ############################################################


    for NUM_CLIENTS, Clist, MAX_ROUNDS, NUM_LOCAL_EPOCHS, lr in zip(num_clients, Cs, max_rounds, num_local_epochs, lrs):
        for C in Clist:
            final_round, fig1, fig2 = federated_sim(num_clients=NUM_CLIENTS,
                                            frac_participants=C,
                                            max_rounds=MAX_ROUNDS,
                                            num_local_epochs=NUM_LOCAL_EPOCHS,
                                            lr=lr,
                                            iid = False,
                                            )

            fig1.savefig(f"outputs/figures/0_err_acc_N_{NUM_CLIENTS:d}__C_{C:.2f}__iters_{final_round:d}.png", dpi=200)
            fig2.savefig(f"outputs/figures/0_hist_N_{NUM_CLIENTS:d}__C_{C:.2f}__iters_{final_round:d}.png", dpi=200)

    print("Finished experiment.")