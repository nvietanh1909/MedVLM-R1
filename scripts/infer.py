import argparse
import os
import sys

import torch
import yaml
from PIL import Image
from unsloth import FastVisionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned vision-language model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (None = use base model)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_path = args.adapter if args.adapter else cfg["model"]["model_path"]
    logger.info(f"Loading model: {model_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    FastVisionModel.for_inference(model)

    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}")
    print(f"Answer:\n{answer}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
