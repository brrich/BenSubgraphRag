#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o webqsp_train.sh.log-%j

module load anaconda/2023a-pytorch
source activate retriever

echo "starting experiment"

export WANDB_API_KEY="ba2696a6f8ac298d5721f772e8a8e434da8675bc"
WANDB_MODE=offline

python train_uncertain.py -d webqsp

echo "done"
