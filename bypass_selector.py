import os
import torch
import numpy as np
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
import matplotlib.pyplot as plt
from PIL import Image
import json

class VQABypassPredictor(PredictorModeHandler):
    """Predictor that bypasses the selector component"""
    
    def predict_raw(self, model, image_path, question):
        """Get raw VQA predictions without using the selector component"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Question: {question}")

        # Get the sample using the dataset function directly
        sample = get_sample(self._paths_cfg, image_path, question)
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        vocab = sample["vocab"]
        
        # Get index-to-answer mapping from vocab
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        model.eval()
        with torch.no_grad():
            # Forward pass through the model
            output = model(image=image, question=question_ids, padding_mask=mask)
            
            # Extract VQA logits directly
            if hasattr(output, 'logits'):
                vqa_logits = output.logits
            else:
                vqa_logits = output
                
            # Print shapes for debugging
            print(f"[DEBUG] VQA logits shape: {vqa_logits.shape}")
            
            # Get probabilities and predictions
            probs = torch.softmax(vqa_logits, dim=-1)[0]  # First batch item
            
            # Get top k predictions (dynamically set k based on tensor size)
            k = min(5, probs.size(0))  # Get top 5 or fewer if tensor is smaller
            if k > 0:
                top_values, top_indices = torch.topk(probs, k)
                
                # Display results
                print(f"\n--- Top {k} VQA Predictions (bypassing selector) ---")
                for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                    answer = idx_to_answer.get(idx.item(), f"unknown-{idx.item()}")
                    print(f"{i+1}. {answer}: {value.item():.4f}")
                    
                # Get main prediction
                pred_idx = top_indices[0].item()
                main_answer = idx_to_answer.get(pred_idx, f"unknown-{pred_idx}")
                confidence = top_values[0].item()
            else:
                print("\nNo predictions available - logits tensor is empty")
                main_answer = "unknown"
                confidence = 0.0
            
            print(f"\n[RESULT] Raw VQA Answer: {main_answer} (Confidence: {confidence:.4f})")
            
            # Display the image with prediction
            plt.figure(figsize=(10, 8))
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"Q: {question}\nA: {main_answer} ({confidence:.4f})")
            plt.axis('off')
            plt.show()
            
            return main_answer, confidence

def compare_predictions(image_path, question):
    """Compare standard predictions (with selector) to raw VQA predictions"""
    # Initialize predictors
    standard_predictor = PredictorModeHandler()
    bypass_predictor = VQABypassPredictor()
    
    # Load the model
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
        
    print(f"Loading model from {model_path}...")
    model = standard_predictor.load_final_model(model_path)
    
    # Make predictions
    print("\n=== Standard Prediction (with selector) ===")
    standard_predictor.predict_sample(model, image_path, question)
    
    print("\n=== Raw VQA Prediction (bypassing selector) ===")
    bypass_predictor.predict_raw(model, image_path, question)

def main():
    print("=== VQA Model Testing (Bypassing Selector) ===")
    
    # Get image path
    image_path = "example/example.png"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
        
    # Display the image
    plt.figure(figsize=(8, 6))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Test Image")
    plt.show()
    
    # Test questions
    questions = [
        "Ảnh này có gì?",  # What's in this image?
        "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
        "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
        "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
    ]
    
    # Run comparisons for each question
    for question in questions:
        print("\n" + "="*50)
        compare_predictions(image_path, question)
        
    # Interactive mode
    print("\n" + "="*50)
    print("\nEnter your own questions (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
        compare_predictions(image_path, question)

if __name__ == "__main__":
    main() 