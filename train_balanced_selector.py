import os
import torch
import yaml
from utils.config import ConfigLoader
from main import train as original_train
import models.model
from utils.trainer import TrainingModeHandler
from transformers import TrainingArguments, Trainer, TrainerCallback
import mlflow

def update_model_config():
    """Update model configuration to make the selector less conservative"""
    # Read the current config
    model_config_path = 'configs/model.yaml'
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Enable the selector
    model_config['use_selector'] = True
    
    # Adjust selector parameters to be less conservative
    # 1. Update the loss weighting to penalize abstention more 
    model_config['selector']['abstention_penalty'] = 2.0  # Higher penalty for abstaining
    
    # 2. Modify other parameters that affect the selector's behavior
    model_config['selector']['params']['use_softmax'] = True  # Enable softmax for more balanced predictions
    
    # Save the updated configuration
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    print("Updated model configuration to make selector less conservative")
    return model_config

def update_training_config():
    """Update training configuration for balanced selector training"""
    # Read the current config
    config_path = 'configs/training.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Adjust training parameters for better selector training
    config['learning_rate'] = 5e-5  # Slightly higher learning rate
    config['weight_decay'] = 0.03   # More regularization
    config['epochs'] = 20           # Enough epochs to learn but avoid overfitting
    
    # Save the updated configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Updated training configuration for balanced selector training")
    return config

# Custom TrainerCallback to adjust the loss function to penalize abstention more
class BalancedSelectorCallback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.abstention_penalty = cfg.get("model").get("selector", {}).get("abstention_penalty", 1.0)
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self.cfg.get("model")["use_selector"]:
            print(f"Selector is ON with abstention penalty {self.abstention_penalty}.")
            print("Computing balanced selector loss to avoid excessive abstention.")
        else:
            print("Selector is OFF. Only getting logits from VQA model and computing VQA loss.")

# Custom modification of the compute_loss function in the model
def patch_model_compute_loss():
    """Monkey patch the model's compute_loss method to penalize abstention more"""
    original_compute_loss = models.model.BEiT3Wrapper.compute_loss
    
    def balanced_compute_loss(self, logits, labels=None, confidences=None, use_selector=None):
        """Modified compute_loss function that penalizes abstention more"""
        loss = None
        
        if not use_selector:
            # Standard loss for VQA model
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, ignore_index=self.unk_index)
        else:
            # Modified loss for selector with abstention penalty
            if labels is not None:
                # Get correctness labels (1 if correct, 0 if incorrect)
                pred_inds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
                correctness = (pred_inds == labels).to(dtype=torch.long)
                
                # Valid labels only (excluding unknown)
                valid_mask = labels != self.unk_index
                if valid_mask.any():
                    # Get filtered confidences and correctness
                    filtered_confidences = confidences[valid_mask]
                    filtered_correctness = correctness[valid_mask]
                    
                    # Add an abstention penalty: apply higher weight to abstention errors
                    # This encourages the model to abstain less
                    abstention_penalty = getattr(self, 'abstention_penalty', 2.0)
                    
                    # Create weights: higher weight for misclassifying correctness as incorrect
                    weights = torch.ones_like(filtered_correctness, dtype=torch.float)
                    weights[filtered_correctness == 1] = abstention_penalty  # Higher penalty for not answering when it should
                    
                    # Apply weighted cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        filtered_confidences, filtered_correctness, weight=weights)
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return models.model.ViVQAOutput(
            loss=loss,
            logits=confidences if use_selector else logits
        )
    
    # Replace the original method
    models.model.BEiT3Wrapper.compute_loss = balanced_compute_loss
    models.model.BEiT3Wrapper.abstention_penalty = 2.0  # Add the penalty attribute
    
    print("Patched model's compute_loss function to penalize abstention more")

class BalancedTrainingModeHandler(TrainingModeHandler):
    """Modified handler that creates a more balanced dataset for training"""
    
    def build_compute_metrics(self):
        """Override compute_metrics to provide more insight into selector behavior"""
        orig_compute_metrics = super().build_compute_metrics()
        
        def enhanced_compute_metrics(p):
            # Get original metrics
            metrics = orig_compute_metrics(p)
            
            # Add logging
            logits, labels = p
            print(f"Computing metrics on {len(labels)} examples")
            
            if self._use_selector and logits.shape[1] == 2:
                # Binary classification case for selector
                selector_preds = (logits[:, 1] > logits[:, 0]).astype(int)
                print(f"Selector distribution: {sum(selector_preds)} positive, {len(selector_preds) - sum(selector_preds)} negative")
                print(f"This means the selector is answering {sum(selector_preds)/len(selector_preds)*100:.2f}% of questions")
            
            return metrics
            
        return enhanced_compute_metrics

def train_balanced_selector():
    """Run training with modified parameters for a less conservative selector"""
    # Update configurations
    update_model_config()
    update_training_config()
    
    # Patch model's compute_loss method
    patch_model_compute_loss()
    
    # Override the TrainingModeHandler with our balanced version
    original_handler = TrainingModeHandler
    models.model.TrainingModeHandler = BalancedTrainingModeHandler
    
    # Run the original training function
    print("Starting training with balanced selector configuration...")
    original_train()
    
    # Restore original handler
    models.model.TrainingModeHandler = original_handler
    
    print("Training completed. The selector should now be less conservative.")

if __name__ == '__main__':
    train_balanced_selector() 