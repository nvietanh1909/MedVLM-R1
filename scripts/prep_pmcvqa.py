import os
import pandas as pd
import json
import re

def process_pmcvqa(input_csv, output_json, image_dir):
    df = pd.read_csv(input_csv)
    data = []
    
    bad_phrases = [
        "none of the above",
        "all of the above",
        "both a and b",
        "both b and c",
        "both a and c",
        "cannot determine",
        "cannot be determined",
        "a and b are correct"
    ]
    
    for _, row in df.iterrows():
        ans_label = str(row.get('Answer_label', '')).strip()
        choice_col = f"Choice {ans_label}"
        
        if choice_col in df.columns:
            gt_text = str(row[choice_col]).strip()
        else:
            continue
            
        gt_text = re.sub(r'^[A-D]\s*:\s*', '', gt_text).strip()
        
        gt_lower = gt_text.lower()
        if any(bad in gt_lower for bad in bad_phrases):
            continue
            
        img_path = os.path.join(image_dir, str(row['Figure_path']))
        
        data.append({
            "image": img_path,
            "question": str(row['Question']).strip(),
            "ground_truth": gt_text
        })
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed {input_csv} -> Kept {len(data)} valid open-ended samples.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset", "PMC-VQA")
    
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    
    train_out = os.path.join(dataset_dir, "train_open_ended.json")
    test_out = os.path.join(dataset_dir, "test_open_ended.json")
    
    img_dir = "dataset/PMC-VQA/images" 
    
    if os.path.exists(train_csv):
        process_pmcvqa(train_csv, train_out, img_dir)
    if os.path.exists(test_csv):
        process_pmcvqa(test_csv, test_out, img_dir)
