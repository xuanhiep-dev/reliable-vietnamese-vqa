import os
import torch
import numpy as np
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
import matplotlib.pyplot as plt
from PIL import Image
import json
import yaml
from omegaconf import OmegaConf
from inference.predictor import MODEL_CONFIG

class VQANoSelectorPredictor(PredictorModeHandler):
    """Predictor that loads model without selector"""
    
    def load_model_without_selector(self, model_path):
        """Load model but force use_selector=False in config"""
        print(f"Loading model from {model_path} with selector disabled...")
        
        # Load config but disable selector
        model_config = OmegaConf.create(MODEL_CONFIG)
        model_config.update({
            'use_selector': False,  # Disable selector
            'model_path': model_path
        })
        
        # Load model with modified config
        model = self.load_model(model_config=model_config)
        return model
    
    def predict_no_selector(self, model, image_path, question):
        """Make prediction with model's selector disabled"""
        device = self._device
        sample = get_sample(self._paths_cfg, image_path, question)
        
        # Process inputs
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        
        # Get vocabulary for converting indices to answers
        vocab = sample["vocab"]
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image=image, question=question_ids, padding_mask=mask)
            
            # Process outputs
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=-1)[0]
            
            # Get top k predictions (dynamically determine k)
            k = min(5, probs.size(0))
            if k > 0:
                top_values, top_indices = torch.topk(probs, k)
                
                # Create result structure
                predictions = []
                for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                    answer = idx_to_answer.get(idx.item(), f"unknown-{idx.item()}")
                    confidence = value.item()
                    predictions.append({
                        "answer": answer,
                        "confidence": confidence
                    })
                
                return {
                    "main_answer": predictions[0]["answer"],
                    "main_confidence": predictions[0]["confidence"],
                    "all_predictions": predictions
                }
            else:
                return {
                    "main_answer": "unknown",
                    "main_confidence": 0.0,
                    "all_predictions": []
                }

def format_prediction(prediction):
    """Format prediction for display"""
    main_answer = prediction["main_answer"]
    confidence = prediction["main_confidence"]
    result = f"{main_answer} ({confidence:.4f})"
    
    all_preds = prediction.get("all_predictions", [])
    if len(all_preds) > 1:
        result += "\nAlternatives: "
        alternatives = [f"{p['answer']} ({p['confidence']:.4f})" for p in all_preds[1:]]
        result += ", ".join(alternatives)
    
    return result

def compare_predictions(standard_predictor, no_selector_predictor, model, model_without_selector, image_path, question):
    """Compare predictions with and without selector"""
    print(f"\nQuestion: {question}")
    
    # Get standard prediction (with selector)
    try:
        standard_pred = standard_predictor.predict(model, image_path, question)
        print(f"Standard prediction: {format_prediction(standard_pred)}")
    except Exception as e:
        print(f"Error with standard prediction: {e}")
        standard_pred = {"main_answer": "Error", "main_confidence": 0.0}
    
    # Get prediction without selector
    try:
        no_selector_pred = no_selector_predictor.predict_no_selector(model_without_selector, image_path, question)
        print(f"No selector prediction: {format_prediction(no_selector_pred)}")
    except Exception as e:
        print(f"Error with no-selector prediction: {e}")
        no_selector_pred = {"main_answer": "Error", "main_confidence": 0.0}
    
    return standard_pred, no_selector_pred

def main():
    # Initialize predictors
    standard_predictor = PredictorModeHandler()
    no_selector_predictor = VQANoSelectorPredictor()
    
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
        print("Model checkpoint not found. Please check your model path.")
        return
    
    # Load models
    model = standard_predictor.load_final_model(model_path)
    model_without_selector = no_selector_predictor.load_model_without_selector(model_path)
    
    # Display example image
    image_path = "example/example.png"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.title("Example Image")
        plt.show()
    else:
        print(f"Example image not found at {image_path}")
        return
    
    # Test with some predefined questions
    questions = [
        "Ảnh này có gì?",  # What's in this image?
        "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
        "Thời tiết trong ảnh này thế nào?",  # How's the weather in this image?
        "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
        "Đây là vào mùa nào trong năm?",  # What season is it in the image?
    ]
    
    for question in questions:
        compare_predictions(standard_predictor, no_selector_predictor, model, model_without_selector, image_path, question)
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter your questions (or 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        compare_predictions(standard_predictor, no_selector_predictor, model, model_without_selector, image_path, question)

if __name__ == "__main__":
    print("=== VQA Predictor - Bypassing Selector Alternative Method ===")
    main() 