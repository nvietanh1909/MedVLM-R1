#!/bin/bash

huggingface-cli download FreedomIntelligence/PubMedVision \
  --repo-type dataset \
  --local-dir dataset/PubMedVision \
  --local-dir-use-symlinks False
