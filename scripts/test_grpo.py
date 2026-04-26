import os
import sys
import torch
from PIL import Image
from unsloth import FastVisionModel
import argparse

def test_inference(model_path, image_path, question):
    print(f"Loading model from: {model_path}...")
    
    # Load model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    sys_msg = "You are an expert medical AI assistant. Analyze the medical image and answer the question accurately."
    user_msg = f"Question: {question}\n\nPlease think step by step and output your clinical reasoning inside <think>...</think> tags, then provide your final answer inside <answer>...</answer> tags."
    
    messages = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_msg},
            ],
        }
    ]
    
    print("Generating response...")
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )
    
    # Decode only the new tokens
    generated_ids = output_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)
    print(f"MODEL RESPONSE:\n{response}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--question", type=str, required=True, help="Question for the model")
    
    args = parser.parse_args()
    test_inference(args.model, args.image, args.question)
