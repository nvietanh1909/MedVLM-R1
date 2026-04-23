#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_NAME="pritamdeka/S-PubMedBert-MS-MARCO"
MODEL_BASENAME=$(basename "$MODEL_NAME")
OUTPUT_DIR="$PROJECT_ROOT/models/$MODEL_BASENAME"

mkdir -p "$OUTPUT_DIR"

hf download "$MODEL_NAME" \
  --local-dir "$OUTPUT_DIR"
