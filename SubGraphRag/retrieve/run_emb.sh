#!/bin/bash
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=32G
#SBATCH -o debug_log.txt
#SBATCH -e debug_err.txt

# Ensure module command is available
source /etc/profile
source /etc/profile.d/modules.sh

# module load anaconda/2023a-pytorch  # Use the correct Anaconda version
module load nccl

source activate gte_large_en_v1-5  # Activate your custom env

# Debugging output
echo "Running on $(hostname)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Torch location: $(python -c 'import torch; print(torch.__file__)' 2>/dev/null || echo "Torch not found")"

python emb.py -d webqsp
echo "done"

