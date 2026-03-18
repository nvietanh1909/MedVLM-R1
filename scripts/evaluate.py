import argparse
import json
import os
import sys

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned vision-language model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (None = use base model)")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    return parser.parse_args()


def load_images(data_path: str, image_paths: list) -> list:
    images = []
    for path in image_paths:
        full_path = os.path.join(data_path, path)
        images.append(Image.open(full_path).convert("RGB"))
    return images


def run_inference(model, tokenizer, images: list, question: str, max_new_tokens: int) -> str:
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": user_content}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    first_image = images[0] if images else None
    inputs = tokenizer(
        first_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            top_p=0.8,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


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

    dataset_cfg = cfg["dataset"]
    json_path = os.path.join(dataset_cfg["data_path"], dataset_cfg["json_file"])
    with open(json_path, "r") as f:
        data = json.load(f)

    eval_data = data[-args.num_samples:]
    logger.info(f"Evaluating on last {len(eval_data)} samples")

    results = []
    for sample in tqdm(eval_data, desc="Evaluating"):
        images = load_images(dataset_cfg["data_path"], sample.get("image", []))
        question = sample["conversations"][0]["value"]
        ground_truth = sample["conversations"][1]["value"]

        prediction = run_inference(model, tokenizer, images, question, args.max_new_tokens)

        results.append({
            "id": sample.get("id", ""),
            "modality": sample.get("modality", ""),
            "body_part": sample.get("body_part", ""),
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
        })

    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
