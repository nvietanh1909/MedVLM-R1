import re
import torch
from transformers import pipeline, AutoTokenizer
from src.rewards.utils import extract_answer
from transformers import BitsAndBytesConfig
class LLMJudgeReward:
    SCALE = 1.0

    def __init__(self):
        model_id = "models/deepseek-32b"
        print(f"Loading Local LLM Judge: {model_id} in 4-bit...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
            device_map="auto"
        )

    def __call__(self, completions, question, ground_truth, **kwargs):
        prompts = []
        for comp, q, gt in zip(completions, question, ground_truth):
            ans = extract_answer(comp)
            if len(ans.strip()) < 5:
                prompts.append(None)
                continue
                
            prompt = f"""You are an expert Medical Reasoning Judge. You evaluate the LOGIC and MEDICAL EXPERTISE of an AI's reasoning process.
IMPORTANT: You cannot see the medical image, but the AI can. You are provided with the verified Ground Truth diagnosis.

Question: {q}
Ground Truth (verified by doctors): {gt}

AI Full Response:
{comp}

Your task: Assume the Ground Truth is the absolute truth about the image. Score the LOGICAL QUALITY of the AI's reasoning chain (inside <think> tags) that leads to its conclusion.

Scoring rubric (0.0 to 1.0):
- 0.0 to 0.2: Reasoning contradicts medical science, or has no logical connection to the Ground Truth.
- 0.3 to 0.5: Generic reasoning, just restates the question or gives a conclusion without analyzing medical signs.
- 0.6 to 0.8: Analyzes relevant medical signs logically leading to the Ground Truth, concise writing style.
- 0.9 to 1.0: Sharp reasoning, clearly explains imaging findings/mechanisms that prove the Ground Truth. Penalize verbose/rambling text.

Output ONLY a single number (e.g., 0.8). No <think> tags, no explanation."""
            
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)

        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [p for p in prompts if p is not None]
        
        results = [0.0] * len(completions)
        
        if valid_prompts:
            try:
                outputs = self.pipe(
                    valid_prompts, 
                    max_new_tokens=150, 
                    temperature=0.0, 
                    do_sample=False, 
                    batch_size=8, 
                    return_full_text=False
                )
                for i, out in zip(valid_indices, outputs):
                    score_str = out[0]["generated_text"].strip()
                    score_str = re.sub(r'<think>.*?</think>', '', score_str, flags=re.DOTALL)
                    
                    scores = re.findall(r"[-+]?\d*\.\d+|\d+", score_str)
                    if scores:
                        score = float(scores[-1])
                        results[i] = max(0.0, min(1.0, score)) * self.SCALE
            except Exception as e:
                print(f"LLM Judge error: {e}")
                
        return results
