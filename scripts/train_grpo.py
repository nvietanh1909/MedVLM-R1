import argparse
import os
import sys

import torch
import yaml
import warnings
import unsloth
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.registry import get_dataset
from src.models.loader import load_model
from src.rewards import get_reward_funcs
from src.training.grpo_trainer import build_grpo_trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.max_steps is not None:
        cfg["grpo"]["max_steps"] = args.max_steps
    if args.max_samples is not None:
        cfg["dataset"]["max_samples"] = args.max_samples
    if args.offset is not None:
        cfg["dataset"]["offset"] = args.offset

    cfg["grpo"]["learning_rate"] = float(cfg["grpo"]["learning_rate"])

    logger.info(f"Config: {args.config}")
    model_loc = cfg['model'].get('path', cfg['model'].get('model_path', 'unknown'))
    logger.info(f"Model: {model_loc}")
    ds_loc = cfg['dataset'].get('path', cfg['dataset'].get('json_file', 'unknown'))
    logger.info(f"Dataset: {cfg['dataset']['type']} — {ds_loc}")
    logger.info(f"Output: {cfg['grpo']['output_dir']}")

    model, tokenizer = load_model(cfg)

    dataset = get_dataset(cfg, tokenizer)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    reward_funcs = get_reward_funcs(cfg)
    logger.info(f"Rewards: {[type(r).__name__ for r in reward_funcs]}")

    trainer = build_grpo_trainer(model, tokenizer, dataset, reward_funcs, cfg)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    logger.info(
        f"GPU: {gpu_stats.name} | "
        f"Total VRAM: {round(gpu_stats.total_memory / 1024**3, 2)} GB"
    )
    logger.info(f"Memory reserved before training: {start_mem} GB")

    warnings.filterwarnings("ignore", message=".*Kwargs passed to `processor.__call__`.*")
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*and.*max_length.*")

    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    output_dir = cfg["grpo"]["output_dir"]
    resume_from_checkpoint = None
    
    checkpoints = []
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    if checkpoints:
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        resume_from_checkpoint = latest_checkpoint
        logger.info(f"Auto-resume enabled: Found latest checkpoint at {latest_checkpoint}")
    elif model_loc.startswith(output_dir) and "checkpoint-" in model_loc:
        resume_from_checkpoint = model_loc
        logger.info(f"Smart Resume enabled: Resuming from {resume_from_checkpoint}")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    runtime_min = round(trainer_stats.metrics["train_runtime"] / 60, 2)
    logger.info(f"Training complete — {runtime_min} min | Peak VRAM: {used_mem} GB")

    output_dir = cfg["grpo"]["output_dir"]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved → {output_dir}")


if __name__ == "__main__":
    main()
