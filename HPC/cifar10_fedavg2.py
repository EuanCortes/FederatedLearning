import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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

current_dir = os.path.dirname(os.path.realpath(__file__))   # current directory
parent_dir = os.path.dirname(current_dir)                   # parent directory
model_dir = os.path.join(parent_dir, 'utils')               # model directory

if model_dir not in sys.path:
    sys.path.append(model_dir)  # add model to pythonpath

from model import SmallCNN
from dataset import load_data, split_data
from federated_learning import fed_avg, validate

############################################################
# --------------------- End imports ---------------------- #
############################################################



###########################
# Client Local Update Worker
###########################
def client_update_worker(global_state, device, client_loader, epochs):
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
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
    # Return updated weights on CPU to avoid CUDA tensor sharing issues
    return {k: v.cpu() for k, v in model.state_dict().items()}

###########################
# GPU Process Function
###########################
def gpu_process(gpu_id, client_ids, trainset, client_indices, global_state, epochs, return_dict, max_streams=4):
    """
    Run local client updates on a single GPU.
    - gpu_id: GPU index (e.g., 0, 1)
    - client_ids: list of client indices assigned to this GPU
    - trainset: the full training dataset
    - client_indices: a list (or dict) mapping each client id to its indices in trainset
    - global_state: current global model state (on CPU)
    - epochs: number of local epochs for each client update
    - return_dict: a Manager dict to store client updates
    - max_streams: maximum number of concurrent CUDA streams
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    print(f"GPU {gpu_id} processing clients: {client_ids}", flush=True)
    
    # Create a fixed pool of CUDA streams
    local_updates = {}
    
    # Process clients in batches of size max_streams
    for client_id in client_ids:
        # Recreate DataLoader for this client using its subset of the trainset.
        client_subset = Subset(trainset, client_indices[client_id])
        client_loader = DataLoader(client_subset, batch_size=64, shuffle=True, num_workers=0)
        update = client_update_worker(global_state, device, client_loader, epochs)
        local_updates[client_id] = update

    
    return_dict[gpu_id] = local_updates
    print(f"GPU {gpu_id} finished processing clients: {client_ids}", flush=True)

###########################
# Federated Simulation Function
###########################
def federated_simulation(num_clients, frac_participants, max_rounds, num_local_epochs):
    """
    Main simulation function for federated learning.
    - num_clients: total number of clients
    - frac_participants: fraction of clients selected each round
    - max_rounds: maximum number of rounds to run
    - num_local_epochs: number of local epochs for client updates
    """
    clients_per_round = int(num_clients * frac_participants)
    
    # Load datasets
    trainset, valset, testset = load_data(validation_percent=0.2)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=0)
    
    # Partition the training set among clients.
    # Assume split_data returns a list (or dict) where each index contains the list of indices for that client.
    client_indices = split_data(trainset, num_clients=num_clients, iid=True)
    
    # Initialize global model
    model = SmallCNN()
    global_state = copy.deepcopy(model.state_dict())

    main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(main_device)

    num_gpus = torch.cuda.device_count()
    print(f"Running simulation on {num_gpus} GPUs", flush=True)
    
    # Main federated learning loop
    for rnd in range(max_rounds):
        print(f"\nStarting round {rnd+1}", flush=True)
        selected_clients = random.sample(range(num_clients), clients_per_round)
        print("Selected clients:", selected_clients, flush=True)
        
        # Partition selected clients evenly among available GPUs
        gpu_partitions = {i: [] for i in range(num_gpus)}
        for idx, client_id in enumerate(selected_clients):
            gpu_partitions[idx % num_gpus].append(client_id)
        
        # Use a Manager to collect client updates from each GPU process
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        for gpu_id, client_ids in gpu_partitions.items():
            if client_ids:  # only launch process if there are clients assigned
                p = mp.Process(target=gpu_process, args=(
                    gpu_id, client_ids, trainset, client_indices, global_state, num_local_epochs, return_dict))
                p.start()
                processes.append(p)
        
        for p in processes:
            p.join()
        
        # Gather client updates from all GPUs and perform federated averaging
        client_updates = []
        for gpu_updates in return_dict.values():
            client_updates.extend(gpu_updates.values())
        
        global_state = fed_avg(client_updates)
        print("FedAvg complete", flush=True)
        
        # Evaluate global model on the validation set
        val_acc = validate(model, global_state, val_loader)
        print(f"Round {rnd+1} validation accuracy: {val_acc:.3f}", flush=True)
    
    return global_state

###########################
# Main Execution Block
###########################
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # Hyperparameters
    NUM_CLIENTS = 100
    FRACTION_PARTICIPANTS = 0.1  # 10% of clients participate each round
    MAX_ROUNDS = 10
    NUM_LOCAL_EPOCHS = 5
    
    final_state = federated_simulation(
        num_clients=NUM_CLIENTS,
        frac_participants=FRACTION_PARTICIPANTS,
        max_rounds=MAX_ROUNDS,
        num_local_epochs=NUM_LOCAL_EPOCHS
    )
    
    #torch.save(final_state, "final_global_model.pth")
    print("Simulation complete. Final model saved.", flush=True)
