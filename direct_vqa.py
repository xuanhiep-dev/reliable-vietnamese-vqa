import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
from omegaconf import OmegaConf
from inference.predictor import MODEL_CONFIG
import traceback

class DirectVQAPredictor(PredictorModeHandler):
    """Predictor that directly accesses VQA predictions by monkey-patching the model"""
    
    def load_model_and_patch(self, model_path):
        """Load the model and patch it to bypass the selector"""
        print(f"Loading model from {model_path}...")
        model = self.load_final_model(model_path)
        
        # Store the original model to restore later
        self.original_model = model
        
        # Get the actual model (unwrap from DataParallel if needed)
        self.actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # Check if this is a model with a selector
        if hasattr(self.actual_model, 'use_selector'):
            print(f"Model has selector attribute: {self.actual_model.use_selector}")
            
            # Store the original forward method
            self.original_forward = self.actual_model.forward
            
            # Define a patched forward method that bypasses the selector
            def patched_forward(self_model, image, question, padding_mask, labels=None, **kwargs):
                # We'll call the parent class's forward method if available
                if hasattr(self_model.__class__, '__base__') and hasattr(self_model.__class__.__base__, 'forward'):
                    # Call the parent's forward method to get the VQA output
                    parent_cls = self_model.__class__.__base__
                    results = parent_cls.forward(self_model, image, question, padding_mask, labels, **kwargs)
                    
                    # Return just the VQA logits
                    if isinstance(results, dict) and 'logits' in results:
                        return results
                    return results
                else:
                    # Fall back to original forward but ignore selector
                    old_use_selector = self_model.use_selector
                    self_model.use_selector = False
                    results = self.original_forward(image, question, padding_mask, labels, **kwargs)
                    self_model.use_selector = old_use_selector
                    return results
            
            # Replace the forward method on the model instance
            import types
            self.actual_model.forward = types.MethodType(patched_forward, self.actual_model)
            
            print("Model successfully patched to bypass selector")
        else:
            print("Model doesn't have a selector to bypass")
            
        return model
        
    def restore_model(self):
        """Restore the original forward method"""
        if hasattr(self, 'original_forward') and hasattr(self, 'actual_model'):
            import types
            self.actual_model.forward = types.MethodType(self.original_forward, self.actual_model)
            print("Model restored to original state")
    
    def predict_direct(self, model, image_path, question):
        """Make a prediction using the raw VQA model, bypassing selector"""
        device = self._device
        
        # Get the sample
        sample = get_sample(self._paths_cfg, image_path, question)
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        vocab = sample["vocab"]
        
        # Get index-to-answer mapping from vocab
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(image=image, question=question_ids, padding_mask=mask)
                
                # Extract logits - different models output different formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)[0]
                
                # Get top predictions
                k = min(5, probs.size(0))
                if k > 0:
                    top_values, top_indices = torch.topk(probs, k)
                    
                    # Create prediction list
                    predictions = []
                    for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                        answer = idx_to_answer.get(idx.item(), f"unknown-{idx.item()}")
                        confidence = value.item()
                        predictions.append({
                            "answer": answer,
                            "confidence": confidence,
                            "index": idx.item()
                        })
                        print(f"{i+1}. {answer}: {confidence:.4f}")
                    
                    result = {
                        "main_answer": predictions[0]["answer"],
                        "main_confidence": predictions[0]["confidence"],
                        "all_predictions": predictions
                    }
                else:
                    result = {
                        "main_answer": "unknown",
                        "main_confidence": 0.0,
                        "all_predictions": []
                    }
                
                return result
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                traceback.print_exc()
                return {"main_answer": "error", "main_confidence": 0.0}

def format_result(result):
    """Format a prediction result for display"""
    answer = result["main_answer"]
    confidence = result["main_confidence"]
    
    formatted = f"{answer} ({confidence:.4f})"
    
    # Add alternatives if available
    all_preds = result.get("all_predictions", [])
    if len(all_preds) > 1:
        alternatives = [f"{p['answer']} ({p['confidence']:.4f})" for p in all_preds[1:]]
        formatted += "\nAlternatives: " + ", ".join(alternatives)
    
    return formatted

def display_image_with_qa(image_path, question, answer, confidence):
    """Display an image with the question and answer"""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Q: {question}\nA: {answer} ({confidence:.4f})")
    plt.axis('off')
    plt.show()

def main():
    print("=== Direct VQA Prediction (No Selector) ===")
    
    # Initialize predictor
    predictor = DirectVQAPredictor()
    
    # Find model path
    model_paths = [
        "checkpoints/model_with_selector.pt/checkpoint-150/pytorch_model.bin",
        "checkpoints/model_with_selector.pt/pytorch_model.bin"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Model checkpoint not found. Please specify the model path manually.")
        return
    
    # Load and patch model
    model = predictor.load_model_and_patch(model_path)
    
    # Get example image
    image_path = "example/example.png"
    if not os.path.exists(image_path):
        print(f"Example image not found at {image_path}")
        return
    
    # Display the image
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title("Example Image")
    plt.axis('off')
    plt.show()
    
    # Test with predefined questions
    test_questions = [
        "Ảnh này có gì?",  # What's in this image?
        "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
        "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
        "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
    ]
    
    for question in test_questions:
        print(f"\n--- Question: {question} ---")
        result = predictor.predict_direct(model, image_path, question)
        print(f"Answer: {format_result(result)}")
        
        # Display image with QA
        display_image_with_qa(
            image_path, 
            question, 
            result["main_answer"], 
            result["main_confidence"]
        )
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter your questions (or 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        print(f"\n--- Question: {question} ---")
        result = predictor.predict_direct(model, image_path, question)
        print(f"Answer: {format_result(result)}")
        
        # Display image with QA
        display_image_with_qa(
            image_path, 
            question, 
            result["main_answer"], 
            result["main_confidence"]
        )
    
    # Restore original model
    predictor.restore_model()
    print("Session ended.")

if __name__ == "__main__":
    main() 