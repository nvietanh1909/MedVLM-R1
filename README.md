# Medical VLM Fine-Tuning

This repository contains scripts and configurations for fine-tuning a medical Vision-Language Model (VLM) using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Project Structure

- `configs/`: Model and training configuration files.
- `dataset/`: Storage for downloaded datasets.
- `models/`: Storage for downloaded pre-trained models.
- `scripts/`: Python scripts for training, testing, and downloading resources.
- `*.slurm`: SLURM scripts for running jobs on a cluster.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download all required models and datasets:
```bash
bash setup_all.sh
```
*(Downloads: Qwen3.5-2B, S-PubMedBert-MS-MARCO, DeepSeek-R1-Distill-Qwen-32B, PubMedVision, and PMC-VQA)*

## Usage

### Training (SFT)

Submit the SFT job using SLURM:
```bash
sbatch run_sft.slurm
```

### Testing (GRPO)

Run inference testing on a checkpoint:
```bash
sbatch test_grpo.slurm
```
