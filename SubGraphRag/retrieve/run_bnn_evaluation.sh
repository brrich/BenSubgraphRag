#!/bin/bash

# Script to run BNN inference and evaluation
set -e

# Default values
DATASET="webqsp"
MODEL_PATH=""
MC_SAMPLES=20
MAX_K=500
K_LIST="50,100,200,400"
SAVE_RESULTS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -p|--path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --mc_samples)
      MC_SAMPLES="$2"
      shift 2
      ;;
    --max_k)
      MAX_K="$2"
      shift 2
      ;;
    --k_list)
      K_LIST="$2"
      shift 2
      ;;
    --no-save)
      SAVE_RESULTS=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
  echo "Error: Model path is required (use -p or --path)"
  exit 1
fi

# Check if model path exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file does not exist: $MODEL_PATH"
  exit 1
fi

echo "=== Running BNN evaluation with the following settings ==="
echo "Dataset: $DATASET"
echo "Model path: $MODEL_PATH"
echo "MC samples: $MC_SAMPLES"
echo "Max K for retrieval: $MAX_K"
echo "K list for evaluation: $K_LIST"
echo "Save results: $SAVE_RESULTS"
echo "========================================================"

# Run BNN inference
echo "Running BNN inference..."
python inference_uncertain_bnn.py -p "$MODEL_PATH" --max_K "$MAX_K" --mc_samples "$MC_SAMPLES"

# Get inference result path
DIRNAME=$(dirname "$MODEL_PATH")
RESULT_PATH="$DIRNAME/retrieval_result_bnn_mc${MC_SAMPLES}.pth"

# Check if inference result was created
if [ ! -f "$RESULT_PATH" ]; then
  echo "Error: Inference result not found at: $RESULT_PATH"
  exit 1
fi

# Run evaluation
echo "Running evaluation..."
if [ "$SAVE_RESULTS" = true ]; then
  python eval_uncertain_bnn.py -d "$DATASET" -p "$RESULT_PATH" --k_list "$K_LIST" --save_results
else
  python eval_uncertain_bnn.py -d "$DATASET" -p "$RESULT_PATH" --k_list "$K_LIST"
fi

echo "BNN evaluation completed successfully!" 