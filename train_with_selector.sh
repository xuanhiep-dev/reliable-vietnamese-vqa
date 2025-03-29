#!/bin/bash

# Install wandb if needed
pip install wandb

# You need to get your API key from https://wandb.ai/settings
# Replace YOUR_API_KEY with your actual API key or set it as an environment variable
# If you don't pass anything, you'll be prompted to log in via browser
echo "Logging in to Weights & Biases..."
wandb login '783994d1730df870e08248f8996c52a1d19a370b'

# Initialize wandb project
wandb init -p vietnamese-reliable-vqa

# First run the setup script
python setup_training.py

# Then run the training with explicit paths
python main.py --set training.lyp_mode=false \
                     model.use_selector=true \
                     paths.checkpoints.save_path="checkpoints/model_with_selector.pt" \
                     paths.checkpoints.load_path="checkpoints/pytorch_model.bin" \
                     paths.train_path="data/full/train.csv" \
                     paths.valid_path="data/full/valid.csv" \
                     paths.test_path="data/full/test.csv" 