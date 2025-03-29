import os
import torch
import numpy as np
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
import matplotlib.pyplot as plt
from PIL import Image
import json

class DebugPredictor(PredictorModeHandler):
    """Predictor with enhanced debugging capabilities"""
    
    def debug_model_structure(self, model):
        """Print model structure and parameters"""
        print("\n=== Model Structure Analysis ===")
        print(f"Model type: {type(model)}")
        print(f"Is DataParallel: {isinstance(model, torch.nn.DataParallel)}")
        
        # Get actual model if wrapped in DataParallel
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        print(f"Actual model type: {type(actual_model)}")
        
        # Check for selector
        has_selector = hasattr(actual_model, 'selector')
        print(f"Has selector: {has_selector}")
        if has_selector:
            print(f"Selector type: {type(actual_model.selector)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check for specific attributes
        attrs_to_check = ['use_selector', 'selector', 'head']
        for attr in attrs_to_check:
            if hasattr(actual_model, attr):
                print(f"Has {attr}: {getattr(actual_model, attr)}")
        
        return actual_model
    
    def debug_forward_pass(self, model, image_path, question):
        """Debug the forward pass of the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Debugging forward pass for: {question} ===")
        
        # Get sample
        sample = get_sample(self._paths_cfg, image_path, question)
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        
        # Get vocabulary 
        vocab = sample["vocab"]
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        # Print input shapes
        print(f"Image shape: {image.shape}")
        print(f"Question shape: {question_ids.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # Run forward pass with hooks
        hooks = []
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook
        
        # Register hooks for key components if possible
        model_to_hook = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # Try to register hooks on common components
        try:
            if hasattr(model_to_hook, 'head'):
                hooks.append(model_to_hook.head.register_forward_hook(get_activation('head')))
            
            if hasattr(model_to_hook, 'selector') and hasattr(model_to_hook.selector, 'module'):
                hooks.append(model_to_hook.selector.module.register_forward_hook(get_activation('selector')))
        except Exception as e:
            print(f"Error registering hooks: {e}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                print("Running forward pass...")
                output = model(image=image, question=question_ids, padding_mask=mask)
                
                # Print output type and structure
                print(f"Output type: {type(output)}")
                if hasattr(output, 'logits'):
                    print(f"Logits shape: {output.logits.shape}")
                    if output.logits.shape[1] == 2:
                        print("This appears to be selector output (binary classification)")
                        print(f"Selector confidence values: {output.logits[0]}")
                        # Try to get the actual VQA prediction
                        print("\nAttempting to extract VQA predictions...")
                    else:
                        print("This appears to be VQA output (multi-class prediction)")
                        # Get top predictions
                        logits = output.logits
                        probs = torch.softmax(logits, dim=-1)[0]
                        k = min(5, probs.size(0))
                        
                        if k > 0:
                            top_values, top_indices = torch.topk(probs, k)
                            print(f"\nTop {k} predictions:")
                            for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                                answer = idx_to_answer.get(idx.item(), f"unknown-{idx.item()}")
                                print(f"{i+1}. {answer}: {value.item():.4f}")
                
                # Check activations captured by hooks
                print("\nActivations captured:")
                for name, act in activation.items():
                    if isinstance(act, torch.Tensor):
                        print(f"{name} shape: {act.shape}")
                    else:
                        print(f"{name} type: {type(act)}")
                        if isinstance(act, dict) and 'confidences' in act:
                            print(f"Confidences shape: {act['confidences'].shape}")
                
                return output
                
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Remove hooks
                for hook in hooks:
                    hook.remove()

def analyze_model(model_path=None):
    """Analyze model structure and behavior"""
    # Initialize predictor
    predictor = DebugPredictor()
    
    # Find model path
    if model_path is None:
        model_paths = [
            "checkpoints/model_with_selector.pt/checkpoint-150/pytorch_model.bin",
            "checkpoints/model_with_selector.pt/pytorch_model.bin"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path or not os.path.exists(model_path):
        print("Model checkpoint not found. Please specify the model path manually.")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = predictor.load_final_model(model_path)
    
    # Debug model structure
    actual_model = predictor.debug_model_structure(model)
    
    # Debug forward pass
    image_path = "example/example.png"
    if not os.path.exists(image_path):
        print(f"Example image not found at {image_path}")
        return
    
    questions = [
        "Ảnh này có gì?",  # What's in this image?
        "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
    ]
    
    for question in questions:
        output = predictor.debug_forward_pass(model, image_path, question)

    print("\n=== Trying to bypass selector ===")
    # Try to directly get VQA logits by modifying the model if possible
    try:
        # Save original forward method
        if hasattr(actual_model, 'forward'):
            original_forward = actual_model.forward
            
            # Replace forward method to bypass selector
            def vqa_forward_only(self, image, question, padding_mask, labels=None, **kwargs):
                # Call the base model forward method but stop before selector
                if hasattr(actual_model, 'beit3'):
                    outputs = self.beit3(
                        textual_tokens=question,
                        visual_tokens=image,
                        text_padding_position=padding_mask
                    )
                    
                    x = outputs["encoder_out"]
                    if hasattr(self, 'pooler'):
                        cls_rep = self.pooler(x)
                        if hasattr(self, 'head'):
                            logits = self.head(cls_rep)
                            return {"logits": logits, "outputs": outputs}
                
                # Fallback to original
                results = original_forward(self, image, question, padding_mask, labels, **kwargs)
                return results
            
            # Try to patch the model
            setattr(type(actual_model), '_temp_forward', vqa_forward_only)
            
            # Test with patched forward
            temp_model = model
            temp_model.eval()
            
            # Get sample
            sample = get_sample(predictor._paths_cfg, image_path, questions[0])
            image = sample["image"].to(predictor._device, dtype=torch.float32)
            question_ids = sample["question"].to(predictor._device)
            mask = sample["padding_mask"].to(predictor._device)
            
            # Try the patched forward
            with torch.no_grad():
                print("Attempting to use patched forward method to bypass selector...")
                if hasattr(actual_model, '_temp_forward'):
                    patched_output = actual_model._temp_forward(
                        actual_model, 
                        image, 
                        question_ids, 
                        mask
                    )
                    print(f"Patched output type: {type(patched_output)}")
                    if isinstance(patched_output, dict) and 'logits' in patched_output:
                        print(f"Logits shape: {patched_output['logits'].shape}")
            
            # Restore original
            if hasattr(type(actual_model), '_temp_forward'):
                delattr(type(actual_model), '_temp_forward')
            
    except Exception as e:
        print(f"Error during patching attempt: {e}")

if __name__ == "__main__":
    print("=== Model Debugging Tool ===")
    analyze_model() 