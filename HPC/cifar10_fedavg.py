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
def client_update_worker(global_state, device, client_loader, epochs, stream):
    """
    Perform a local update for a single client.
    - global_state: global model state (assumed to be on CPU)
    - device: target GPU device
    - client_loader: DataLoader for the client's local data
    - epochs: number of local epochs
    - stream: CUDA stream for asynchronous execution
    Returns a state_dict with parameters moved to CPU.
    """
    model = SmallCNN().to(device)
    model.load_state_dict(global_state)  # load global weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for _ in range(epochs):
        for X, Y in client_loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            # Execute on the provided stream
            with torch.cuda.stream(stream):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, Y)
                loss.backward()
                optimizer.step()
    # Return updated weights on CPU to avoid CUDA tensor sharing issues
    return {k: v.cpu() for k, v in model.state_dict().items()}

###########################
# Federated Simulation Function (Single GPU)
###########################
def federated_simulation(num_clients, frac_participants, max_rounds, num_local_epochs, max_streams=4):
    """
    Main simulation function for federated learning on a single GPU.
    - num_clients: total number of clients
    - frac_participants: fraction of clients selected each round
    - max_rounds: maximum number of rounds to run
    - num_local_epochs: number of local epochs for client updates
    - max_streams: maximum number of concurrent CUDA streams (i.e. client updates in parallel)
    """
    clients_per_round = int(num_clients * frac_participants)
    
    # Load datasets
    trainset, valset, testset = load_data(validation_percent=0.2)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=0)
    
    # Partition the training set among clients.
    # Assume split_data returns a list (or dict) where each index contains the indices for that client.
    client_indices = split_data(trainset, num_clients=num_clients, iid=True)
    
    # Initialize global model and state
    model = SmallCNN()
    global_state = copy.deepcopy(model.state_dict())
    
    # Use a single GPU device (cuda:0 if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running simulation on device: {device}", flush=True)
    
    # Create a fixed pool of CUDA streams
    streams = [torch.cuda.Stream(device) for _ in range(max_streams)]
    
    # Main federated learning loop
    for rnd in range(max_rounds):
        print(f"\nStarting round {rnd+1}", flush=True)
        selected_clients = random.sample(range(num_clients), clients_per_round)
        print("Selected clients:", selected_clients, flush=True)
        
        local_updates = []
        
        # Process selected clients in batches of size max_streams.
        for i in range(0, len(selected_clients), max_streams):
            batch_ids = selected_clients[i:i+max_streams]
            # Launch client updates concurrently for the batch.
            for idx, client_id in enumerate(batch_ids):
                # Recreate DataLoader for this client using its subset of the trainset.
                client_subset = Subset(trainset, client_indices[client_id])
                client_loader = DataLoader(client_subset, batch_size=64, shuffle=True, num_workers=0)
                update = client_update_worker(global_state, device, client_loader, num_local_epochs, streams[idx])
                local_updates.append(update)
            # Wait until all operations in the current batch have finished.
            torch.cuda.synchronize(device)
        
        # Perform federated averaging to update the global model
        global_state = fed_avg(local_updates)
        print("FedAvg complete", flush=True)
        
        # Evaluate global model on the validation set
        model.load_state_dict(global_state)
        val_acc = validate(model, global_state, val_loader)
        print(f"Round {rnd+1} validation accuracy: {val_acc:.3f}", flush=True)
    
    return global_state

###########################
# Main Execution Block
###########################
if __name__ == '__main__':
    # Hyperparameters
    NUM_CLIENTS = 100
    FRACTION_PARTICIPANTS = 0.1  # 10% of clients participate each round
    MAX_ROUNDS = 50
    NUM_LOCAL_EPOCHS = 5
    max_steams_ls = [1, 2, 4, 8, 16]

    for max_streams in max_steams_ls:
        print(f"Running simulation with {max_streams} concurrent streams", flush=True)

        t1 = time.perf_counter()
        final_state = federated_simulation(
            num_clients=NUM_CLIENTS,
            frac_participants=FRACTION_PARTICIPANTS,
            max_rounds=MAX_ROUNDS,
            num_local_epochs=NUM_LOCAL_EPOCHS,
            max_streams=max_streams,  # Limit to 4 concurrent CUDA streams
        )
        t2 = time.perf_counter()
        print(f"Simulation completed in {t2-t1:.2f} seconds\n", flush=True)

        
    # Optionally, save the final global model state
    # torch.save(final_state, "final_global_model.pth")
    #print("Simulation complete. Final model saved.", flush=True)
