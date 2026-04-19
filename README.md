# Medical Vision-Language Model Fine-Tuning Pipeline

This project implements a complete two-stage fine-tuning pipeline for Vision-Language Models (specifically Qwen3.5-2B) on medical datasets (PubMedVision), focusing on creating a model with reasoning capabilities (Thinking Mode) via Group Relative Policy Optimization (GRPO).

## Project Structure

```text
finetune/
├── configs/
│   ├── qwen3.5-2b_pubmedvision.yaml            # Config for Stage 1: SFT
│   └── qwen3.5-2b_pubmedvision_grpo.yaml       # Config for Stage 2: GRPO
├── src/
│   ├── data/
│   │   ├── pubmedvision.py                     # SFT dataset loader
│   │   ├── pubmedvision_grpo.py                # GRPO dataset loader
│   │   └── registry.py                         # Dataset registry
│   ├── models/
│   │   └── loader.py                           # Smart Model + LoRA adapter loader
│   ├── rewards/                                # GRPO Reward Functions
│   │   ├── format_reward.py                    # Ensures <think> and <answer> tags
│   │   ├── correctness_reward.py               # ROUGE-L scoring against Ground Truth
│   │   ├── semantic_reward.py                  # Cosine Similarity using SentenceTransformers
│   │   ├── medical_entity_reward.py            # Validates correct Modality and Body Part
│   │   └── quality_reward.py                   # Length penalty and Anti-hallucination
│   ├── training/
│   │   ├── trainer.py                          # SFTTrainer builder
│   │   └── grpo_trainer.py                     # GRPOTrainer builder
│   └── prompts.py                              # System prompts governing Thinking Mode
├── scripts/
│   ├── train.py                                # Entry point for SFT
│   ├── train_grpo.py                           # Entry point for GRPO
│   ├── evaluate.py                             # Evaluation script
│   └── infer.py                                # Interactive inference
├── run_sft.slurm                               # SLURM script for Stage 1
└── run_grpo.slurm                              # SLURM script for Stage 2
```

## Setup and Installation

Install the required libraries. If you are behind a proxy, remember to bypass it if necessary.

```bash
pip install unsloth transformers==5.3.0
pip install --no-deps trl==0.22.2
pip install -r requirements.txt
```

## The Two-Stage Training Pipeline

This project is intended to be executed on a SLURM-managed cluster across two distinct phases. 

### Stage 1: Supervised Fine-Tuning (SFT)
This builds the foundational capability, teaching the model to understand medical images, align vocabulary with vision, and act as a medical assistant.

Submit the job:
```bash
sbatch run_sft.slurm
```
This will read from `configs/qwen3.5-2b_pubmedvision.yaml` and save the LoRA adapter to `outputs/qwen3.5-2b_pubmedvision`.

### Stage 2: Reinforcement Learning (GRPO)
After SFT completes, the model is trained to "think before speaking". It generates multiple reasoning paths, and our 5-signal reward system evaluates these paths to optimize logic and minimize hallucinations.

Submit the job:
```bash
sbatch run_grpo.slurm
```
This reads from `configs/qwen3.5-2b_pubmedvision_grpo.yaml`. The trainer automatically detects the SFT adapter to build upon it, saving the final intelligent model to `outputs/qwen3.5-2b_pubmedvision_grpo`.

## Quick Testing

If you want to quickly test the pipeline locally on a subset before running the full SLURM jobs, use the command line overrides:

For SFT:
```bash
python scripts/train.py \
    --config configs/qwen3.5-2b_pubmedvision.yaml \
    --max_steps 50 \
    --max_samples 1000
```

For GRPO:
```bash
python scripts/train_grpo.py \
    --config configs/qwen3.5-2b_pubmedvision_grpo.yaml \
    --max_steps 100 \
    --max_samples 500
```

## Inference

Run interactive inference using the final GRPO checkpoints. The model will stream its thought progress if requested.

```bash
python scripts/infer.py \
    --config configs/qwen3.5-2b_pubmedvision_grpo.yaml \
    --adapter outputs/qwen3.5-2b_pubmedvision_grpo \
    --image /path/to/image.jpg \
    --question "Analyze this medical image." \
    --thinking
```

## Evaluation

Evaluate accuracy on a test slice of the dataset:

```bash
python scripts/evaluate.py \
    --config configs/qwen3.5-2b_pubmedvision_grpo.yaml \
    --adapter outputs/qwen3.5-2b_pubmedvision_grpo \
    --num_samples 200 \
    --output_file eval_results.json
```

## Hardware and VRAM Guidelines

Using Unsloth heavily reduces VRAM consumption. The numbers below reflect single GPU requirements for small-scale batches.

| Configuration | Estimated VRAM |
|--------|----------|
| 4bit LoRA (r=16) SFT | ~8 GB |
| 16bit LoRA (r=16) SFT | ~12 GB |
| 16bit LoRA (r=64) SFT | ~14 GB |
| 16bit LoRA GRPO (b=1, n=8) | ~20 - 40 GB (Varies by image resolution and sequence length) |

For GPU clusters like H100s, max sequence lengths and completion lengths can be reliably extended in the configuration parameters without severe out-of-memory risks.
