#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 1. Download PubMedVision
DATASET_1="FreedomIntelligence/PubMedVision"
DIR_1="$PROJECT_ROOT/dataset/$(basename "$DATASET_1")"
echo "--- Downloading $DATASET_1 to $DIR_1 ---"
mkdir -p "$DIR_1"
hf download "$DATASET_1" --repo-type dataset --local-dir "$DIR_1"

# 2. Download PMC-VQA
DATASET_2="RadGenome/PMC-VQA"
DIR_2="$PROJECT_ROOT/dataset/$(basename "$DATASET_2")"
echo "--- Downloading $DATASET_2 to $DIR_2 ---"
mkdir -p "$DIR_2"
hf download "$DATASET_2" --repo-type dataset --local-dir "$DIR_2"

echo "All datasets downloaded successfully!"