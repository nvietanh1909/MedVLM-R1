#!/bin/bash
#SBATCH --job-name=qwen_grpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output="logs/slurm-%j-%x.log"

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "Starting GRPO Training..."
python scripts/train_grpo.py --config configs/qwen3.5-2b_pubmedvision_grpo.yaml
echo "GRPO completed."
