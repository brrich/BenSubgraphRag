#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o bnn_inference_test.sh.log-%j
#SBATCH -e debug_err.txt

module load anaconda/2023a-pytorch
source activate retriever

echo "starting experiment"

# Run BNN inference test script
python bnn_inf_test.py --model_path trained_models/webqsp_Apr15_bnn3_working/cpt.pth --sample_id 0 --mc_samples 10 --top_k 10

echo "experiment completed"
