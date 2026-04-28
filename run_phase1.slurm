#!/bin/bash
#SBATCH --job-name=qwen_grpo_phase1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output="logs/slurm-%j-%x.log"

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu/stubs:$LIBRARY_PATH

echo "Starting GRPO Phase 1 Training..."
python scripts/train_grpo.py --config configs/grpo_v1_pubmed.yaml
echo "GRPO Phase 1 completed."
