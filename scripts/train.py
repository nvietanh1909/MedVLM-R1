import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.registry import get_dataset
from src.models.loader import load_model
from src.training.trainer import build_trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a vision-language model with Unsloth")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max_samples for quick testing")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["learning_rate"] = float(cfg["training"]["learning_rate"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])

    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {cfg['model']['model_path']}")
    logger.info(f"Dataset: {cfg['dataset']['type']} — {cfg['dataset']['json_file']}")
    logger.info(f"Output: {cfg['training']['output_dir']}")

    model, tokenizer = load_model(cfg)

    dataset = get_dataset(cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    trainer = build_trainer(model, tokenizer, dataset, cfg)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    logger.info(f"GPU: {gpu_stats.name} | Total VRAM: {round(gpu_stats.total_memory / 1024**3, 2)} GB")
    logger.info(f"Memory reserved before training: {start_mem} GB")

    trainer_stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    runtime_min = round(trainer_stats.metrics["train_runtime"] / 60, 2)
    logger.info(f"Training complete — {runtime_min} minutes | Peak VRAM: {used_mem} GB")

    output_dir = cfg["training"]["output_dir"]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Adapter saved → {output_dir}")


if __name__ == "__main__":
    main()
