import json
import random
from datasets import Dataset

class PMCVQAGRPODataset:
    def __init__(self, data_path, tokenizer, max_samples=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        if max_samples is not None and max_samples < len(raw_data):
            random.seed(42)
            raw_data = random.sample(raw_data, max_samples)
            
        self.data = []
        for item in raw_data:
            sys_msg = "You are an expert medical AI assistant. Analyze the medical image and answer the question accurately."
            user_msg = f"Question: {item['question']}\n\nPlease think step by step and output your clinical reasoning inside <think>...</think> tags, then provide your final answer inside <answer>...</answer> tags."
            
            prompt = [
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": user_msg},
                    ],
                }
            ]
            
            self.data.append({
                "prompt": prompt,
                "ground_truth": item.get("answer", item.get("ground_truth", "")),
                "question": item.get("question", "")
            })
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.data)))
            return [self.data[i] for i in indices]
        return self.data[idx]
