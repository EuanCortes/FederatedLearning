#!/bin/bash

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J cifar10_fedavg

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00

# request system-memory
#BSUB -R "rusage[mem=16GB]"


#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/94/5/156250/Documents/FederatedLearning/FederatedLearning/HPC/outputs/cifar10_fedavg_%J.out
#BSUB -e /zhome/94/5/156250/Documents/FederatedLearning/FederatedLearning/HPC/outputs/cifar10_fedavg_%J.err
# -- end of LSF options --

# module load python3/3.12.4
source /zhome/94/5/156250/Documents/FederatedLearning/.venv/bin/activate

python3 cifar10_fedavg.py 100 0.1 100 5