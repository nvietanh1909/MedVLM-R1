#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 1. Download Main Model
MODEL_1="unsloth/Qwen3.5-2B"
DIR_1="$PROJECT_ROOT/models/$(basename "$MODEL_1")"
echo "--- Downloading Main Model: $MODEL_1 to $DIR_1 ---"
mkdir -p "$DIR_1"
hf download "$MODEL_1" --local-dir "$DIR_1"

# 2. Download Reward Model (Sentence Transformer)
MODEL_2="pritamdeka/S-PubMedBert-MS-MARCO"
DIR_2="$PROJECT_ROOT/models/$(basename "$MODEL_2")"
echo "--- Downloading Reward Model: $MODEL_2 to $DIR_2 ---"
mkdir -p "$DIR_2"
hf download "$MODEL_2" --local-dir "$DIR_2"

# 3. Download Local LLM Judge Model
MODEL_3="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DIR_3="$PROJECT_ROOT/models/$(basename "$MODEL_3")"
echo "--- Downloading LLM Judge Model: $MODEL_3 to $DIR_3 ---"
mkdir -p "$DIR_3"
hf download "$MODEL_3" --local-dir "$DIR_3"

echo "All models downloaded successfully!"
