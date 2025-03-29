import yaml
import os

def update_training_config():
    # Update training config
    config_path = 'configs/training.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Training parameters
    config['epochs'] = 100
    config['train_batch_size'] = 64  # Increased for RTX 4090
    config['eval_batch_size'] = 64   # Increased for RTX 4090
    config['learning_rate'] = 3e-5
    config['weight_decay'] = 0.01
    config['workers'] = 4  # Increased for faster data loading

    # Checkpoint saving configuration
    config['save_strategy'] = "epoch"  # Save at the end of each epoch
    config['save_total_limit'] = 1  # Keep only the best model
    config['save_steps'] = -1  # Disable saving during training steps
    config['save_on_each_node'] = False  # Don't save on each node in distributed training
    
    # Evaluation and model selection
    config['load_best_model_at_end'] = True  # Load the best model at the end of training
    config['metric_for_best_model'] = "accuracy"  # Use accuracy to determine the best model
    config['greater_is_better'] = True  # Higher accuracy is better
    config['evaluation_strategy'] = "epoch"  # Evaluate at the end of each epoch
    config['eval_steps'] = -1  # Disable evaluation during training steps
    config['save_best_model'] = True  # Save only the best model
    config['save_last_model'] = False  # Don't save the last model
    
    # Logging & Reporting
    config['report_to'] = "wandb"  # Report to Weights & Biases
    config['logging_strategy'] = "steps"  # Log metrics at each step
    config['logging_steps'] = 100  # Log every 100 steps
    config['logging_first_step'] = True  # Log the first step metrics
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Updated training configuration")

def update_model_config():
    # Update model config
    model_config_path = 'configs/model.yaml'
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    model_config['use_selector'] = True
    model_config['selector']['freeze_vqa'] = True  # Keep VQA weights frozen initially

    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    print("Updated model configuration")

def update_paths_config():
    # Update paths config
    paths_config_path = 'configs/paths.yaml'
    with open(paths_config_path, 'r') as f:
        paths_config = yaml.safe_load(f)

    # Set specific paths for checkpoints
    paths_config['checkpoints']['save_path'] = "checkpoints/model_with_selector.pt"
    paths_config['checkpoints']['load_path'] = "checkpoints/pytorch_model.bin"
    paths_config['checkpoints']['base_model_path'] = "checkpoints/base_model.pth"

    with open(paths_config_path, 'w') as f:
        yaml.dump(paths_config, f, default_flow_style=False)
    print("Updated paths configuration")

if __name__ == "__main__":
    update_training_config()
    update_model_config()
    update_paths_config() 