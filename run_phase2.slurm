#!/bin/bash
#SBATCH --job-name=qwen_grpo_phase2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00:00
#SBATCH --output="logs/slurm-%j-%x.log"

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu/stubs:$LIBRARY_PATH
export TORCH_COMPILE_DISABLE=1


echo "Starting GRPO Phase 2 Training..."
python scripts/train_grpo.py --config configs/grpo_v2_pmcvqa.yaml
echo "GRPO Phase 2 completed."
