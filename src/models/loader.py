import os

from unsloth import FastVisionModel


def _has_adapter(path):
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def load_model(cfg: dict):
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    model, tokenizer = FastVisionModel.from_pretrained(
        model_cfg["model_path"],
        load_in_4bit=model_cfg["load_in_4bit"],
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    if not _has_adapter(model_cfg["model_path"]):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=lora_cfg["finetune_vision_layers"],
            finetune_language_layers=lora_cfg["finetune_language_layers"],
            finetune_attention_modules=lora_cfg["finetune_attention_modules"],
            finetune_mlp_modules=lora_cfg["finetune_mlp_modules"],
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            random_state=lora_cfg["random_state"],
            use_rslora=lora_cfg["use_rslora"],
            loftq_config=None,
        )

    return model, tokenizer
