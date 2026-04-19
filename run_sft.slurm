#!/bin/bash
#SBATCH --job-name=qwen_sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output="logs/slurm-%j-%x.log"

export WANDB_DISABLED=true

echo "Starting Supervised Fine-Tuning (SFT)..."
python scripts/train.py --config configs/qwen3.5-2b_pubmedvision.yaml
echo "SFT completed."
