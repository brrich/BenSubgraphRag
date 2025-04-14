#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o EVAL_RESULTS.log

module load anaconda/2023a-pytorch
source activate retriever

echo "starting experiment"

export WANDB_API_KEY="b1cf013fa15f74b678f33b1c935f969a8fef57ae"
WANDB_MODE=offline

python eval.py -d webqsp -p trained_models/trained_model_webqsp_subgraphrag_default/webqsp_Apr03-02:13:18/retrieval_result.pth

echo "done"
