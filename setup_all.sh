#!/bin/bash

echo "Starting Setup..."

echo "1. Downloading Datasets..."
bash scripts/download_dataset.sh

echo "2. Downloading Models..."
bash scripts/download_model.sh

echo "3. Setup Complete!"
    
