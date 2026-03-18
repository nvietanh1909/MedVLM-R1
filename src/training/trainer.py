import os

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def build_trainer(model, tokenizer, train_dataset, cfg: dict) -> SFTTrainer:
    FastVisionModel.for_training(model)

    train_cfg = cfg["training"]
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    max_steps = train_cfg.get("max_steps", -1)

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        optim=train_cfg["optim"],
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        max_length=train_cfg["max_length"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=train_cfg["report_to"],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        save_total_limit=3,
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=sft_config,
    )

    return trainer
