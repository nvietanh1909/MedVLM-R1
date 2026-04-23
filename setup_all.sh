#!/bin/bash

echo "Downloading Dataset..."
bash scripts/download_dataset.sh

echo "Downloading Base Model..."
bash scripts/download_model.sh

echo "Downloading Reward Model..."
bash scripts/download_reward_model.sh

echo "Setup Complete!"
