import os

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig


def build_grpo_trainer(model, tokenizer, dataset, reward_funcs, cfg):
    FastVisionModel.for_training(model)

    grpo_cfg = cfg["grpo"]
    os.makedirs(grpo_cfg["output_dir"], exist_ok=True)

    max_steps = grpo_cfg.get("max_steps", -1)

    config = GRPOConfig(
        output_dir=grpo_cfg["output_dir"],
        num_generations=grpo_cfg["num_generations"],
        max_prompt_length=grpo_cfg["max_prompt_length"],
        max_completion_length=grpo_cfg["max_completion_length"],
        beta=float(grpo_cfg.get("beta", 0.001)),
        temperature=float(grpo_cfg.get("temperature", 0.7)),
        top_p=float(grpo_cfg.get("top_p", 0.95)),
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=float(grpo_cfg["learning_rate"]),
        warmup_steps=grpo_cfg.get("warmup_steps", 50),
        logging_steps=grpo_cfg.get("logging_steps", 5),
        save_steps=grpo_cfg.get("save_steps", 200),
        save_total_limit=grpo_cfg.get("save_total_limit", 3),
        optim=grpo_cfg.get("optim", "adamw_8bit"),
        bf16=grpo_cfg.get("bf16", True),
        fp16=grpo_cfg.get("fp16", False),
        seed=grpo_cfg.get("seed", 3407),
        report_to=grpo_cfg.get("report_to", "none"),
        reward_weights=grpo_cfg.get("reward_weights", None),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
    )

    return trainer
