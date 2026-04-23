#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_NAME="FreedomIntelligence/PubMedVision"
DATASET_BASENAME=$(basename "$DATASET_NAME")
OUTPUT_DIR="$PROJECT_ROOT/dataset/$DATASET_BASENAME"

mkdir -p "$OUTPUT_DIR"

hf download "$DATASET_NAME" \
  --repo-type dataset \
  --local-dir "$OUTPUT_DIR"