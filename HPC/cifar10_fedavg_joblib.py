import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import random
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import time

from joblib import Parallel, delayed

############################################################
# ------------- Import from utils directory -------------- #
############################################################

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(parent_dir, 'utils')

if model_dir not in sys.path:
    sys.path.append(model_dir)

from model import SmallCNN
from dataset import load_data, split_data
from federated_learning import fed_avg, validate

############################################################
# --------------------- End imports ---------------------- #
############################################################

###########################
# Client Local Update Worker
###########################


def update_client(client_id, trainset, client_indices, global_state, device, num_local_epochs):
    """
    Update a single client:
      - Create the client-specific DataLoader from the training set using its indices.
      - Create a dedicated CUDA stream.
      - Run the client update.
    """
    client_subset = Subset(trainset, client_indices[client_id])
    client_loader = DataLoader(client_subset, batch_size=64, shuffle=True, num_workers=0)
    
    stream = torch.cuda.Stream(device)

    model = SmallCNN().to(device)
    model.load_state_dict(global_state)  # load global weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for _ in range(num_local_epochs):
        for X, Y in client_loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with torch.cuda.stream(stream):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, Y)
                loss.backward()
                optimizer.step()

    # Ensure all operations complete before returning.
    torch.cuda.synchronize(device)
    # Return updated weights on CPU to avoid CUDA tensor sharing issues
    return {k: v.cpu() for k, v in model.state_dict().items()}

###########################
# Federated Simulation using Joblib (Single GPU)
###########################
def federated_simulation_joblib(num_clients, frac_participants, max_rounds, num_local_epochs, n_jobs=4):
    """
    Federated simulation using joblib to parallelize client updates on a single GPU.
    - num_clients: total number of clients.
    - frac_participants: fraction of clients selected each round.
    - max_rounds: number of federated learning rounds.
    - num_local_epochs: number of local epochs per client.
    - n_jobs: number of parallel jobs to run concurrently (limit).
    """
    clients_per_round = int(num_clients * frac_participants)
    
    # Load datasets.
    trainset, valset, testset = load_data(validation_percent=0.2)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=0)
    
    # Partition the training set among clients.
    client_indices = split_data(trainset, num_clients=num_clients, iid=True)
    
    # Initialize global model and state.
    model = SmallCNN()
    global_state = copy.deepcopy(model.state_dict())
    
    # Use a single GPU device (cuda:0 if available).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running simulation on device: {device}", flush=True)
    
    for rnd in range(max_rounds):
        print(f"\nStarting round {rnd+1}", flush=True)
        selected_clients = random.sample(range(num_clients), clients_per_round)
        print("Selected clients:", selected_clients, flush=True)
        
        # Parallelize client updates using joblib.
        local_updates = Parallel(n_jobs=n_jobs)(
            delayed(update_client)(client_id, trainset, client_indices, global_state, device, num_local_epochs)
            for client_id in selected_clients
        )
        
        # Aggregate client updates using federated averaging.
        global_state = fed_avg(local_updates)
        print("FedAvg complete", flush=True)
        
        # Evaluate the updated global model.
        model.load_state_dict(global_state)
        val_acc = validate(model, global_state, val_loader)
        print(f"Round {rnd+1} validation accuracy: {val_acc:.3f}", flush=True)
    
    return global_state

###########################
# Main Execution Block
###########################
if __name__ == '__main__':
    # Hyperparameters.
    NUM_CLIENTS = 100
    FRACTION_PARTICIPANTS = 0.1  # 10% of clients participate each round.
    MAX_ROUNDS = 10
    NUM_LOCAL_EPOCHS = 5
    
    final_state = federated_simulation_joblib(
        num_clients=NUM_CLIENTS,
        frac_participants=FRACTION_PARTICIPANTS,
        max_rounds=MAX_ROUNDS,
        num_local_epochs=NUM_LOCAL_EPOCHS,
        n_jobs=4  # Limit number of parallel jobs; adjust as appropriate.
    )
    
    print("Simulation complete.", flush=True)
