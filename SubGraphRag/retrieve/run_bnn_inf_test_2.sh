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
python bnn_inf_test.py --model_path trained_models/webqsp_Apr15_bnn3_working/cpt.pth --custom_query "Who was jorkin it.... nae nae style" --target_entity "Theodor Seuss Geisel" --mc_samples 5

echo "experiment completed"
