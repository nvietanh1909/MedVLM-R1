#!/bin/bash

huggingface-cli download unsloth/Qwen3.5-2B \
  --local-dir models/Qwen3.5-2B \
  --local-dir-use-symlinks False
