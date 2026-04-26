#!/bin/bash
#SBATCH --job-name=test_qwen_grpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output="logs/test-%j.log"

export PYTHONUNBUFFERED=1

CHECKPOINT="outputs/qwen3.5-2b_pmcvqa_grpo/checkpoint-400"
IMAGE="dataset/PMC-VQA/images/PMC1064097_F1.jpg"
QUESTION="Describe the findings in this medical image and provide a diagnosis."

echo "Starting Inference Test..."
echo "Model: $CHECKPOINT"
echo "Image: $IMAGE"
echo "Question: $QUESTION"

python scripts/test_grpo.py \
    --model "$CHECKPOINT" \
    --image "$IMAGE" \
    --question "$QUESTION"

echo "Inference Test Completed."
