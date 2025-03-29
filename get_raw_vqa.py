import os
import torch
import numpy as np
import json
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
import traceback

class RawVQAPredictor(PredictorModeHandler):
    """Simple predictor to get raw VQA predictions without selector"""
    
    def __init__(self):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def disable_selector(self, model):
        """Disable selector by setting use_selector to False"""
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        if hasattr(actual_model, 'use_selector'):
            print(f"Original use_selector value: {actual_model.use_selector}")
            actual_model.use_selector = False
            print(f"Modified use_selector value: {actual_model.use_selector}")
        else:
            print("Model doesn't have use_selector attribute")
        
        return model
    
    def get_raw_predictions(self, model, image_path, question, top_k=5):
        """Get raw VQA predictions"""
        device = self._device
        
        # Get sample
        sample = get_sample(self._paths_cfg, image_path, question)
        
        # Process inputs
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        vocab = sample["vocab"]
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            try:
                # Forward pass
                outputs = model(image=image, question=question_ids, padding_mask=mask)
                
                # Extract logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)[0]
                
                # Get top-k predictions
                k = min(top_k, probs.size(0))
                if k > 0:
                    top_values, top_indices = torch.topk(probs, k)
                    
                    # Prepare result dictionary
                    predictions = []
                    for i, (value, idx) in enumerate(zip(top_values, top_indices)):
                        answer = idx_to_answer.get(idx.item(), f"unknown-{idx.item()}")
                        confidence = value.item()
                        predictions.append({
                            "answer": answer,
                            "confidence": confidence,
                            "rank": i + 1,
                            "index": idx.item()
                        })
                    
                    return {
                        "question": question,
                        "top_prediction": {
                            "answer": predictions[0]["answer"],
                            "confidence": predictions[0]["confidence"]
                        },
                        "all_predictions": predictions,
                        "status": "success"
                    }
                else:
                    return {
                        "question": question,
                        "top_prediction": {"answer": "unknown", "confidence": 0.0},
                        "all_predictions": [],
                        "status": "no_predictions"
                    }
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                traceback.print_exc()
                return {
                    "question": question,
                    "top_prediction": {"answer": "error", "confidence": 0.0},
                    "all_predictions": [],
                    "status": "error",
                    "error_message": str(e)
                }

def process_questions(questions, image_path=None, model_path=None, save_results=True):
    """Process a list of questions and return/save results"""
    
    # Initialize predictor
    predictor = RawVQAPredictor()
    
    # Find model checkpoint
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
        return {"status": "error", "message": "Model checkpoint not found"}
    
    # Default image path
    if image_path is None:
        image_path = "example/example.png"
        if not os.path.exists(image_path):
            print(f"Default image not found at {image_path}")
            return {"status": "error", "message": f"Image not found at {image_path}"}
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = predictor.load_final_model(model_path)
    
    # Disable selector
    model = predictor.disable_selector(model)
    
    # Process each question
    results = []
    for question in questions:
        print(f"\nProcessing question: {question}")
        result = predictor.get_raw_predictions(model, image_path, question)
        results.append(result)
        
        # Print top prediction
        top_pred = result["top_prediction"]
        print(f"Answer: {top_pred['answer']} ({top_pred['confidence']:.4f})")
    
    # Save results to file if requested
    if save_results:
        output_file = "raw_vqa_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return {"status": "success", "results": results}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Get raw VQA predictions without selector")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--questions", type=str, nargs="+", help="Questions to ask")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Use default questions if none provided
    questions = args.questions
    if not questions:
        questions = [
            "Ảnh này có gì?",  # What's in this image?
            "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
            "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
            "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
        ]
    
    # Process questions
    results = process_questions(
        questions, 
        image_path=args.image, 
        model_path=args.model, 
        save_results=not args.no_save
    )
    
    # Print final status
    print(f"\nStatus: {results['status']}")
    
    # Interactive mode if not using command line arguments
    if not args.questions and not args.image and not args.model:
        print("\n=== Interactive Mode ===")
        print("Enter your questions (or 'exit' to quit):")
        
        interactive_questions = []
        while True:
            question = input("\nYour question: ")
            if question.lower() in ["exit", "quit", "q"]:
                break
            interactive_questions.append(question)
        
        if interactive_questions:
            print("\nProcessing interactive questions...")
            process_questions(
                interactive_questions, 
                image_path=args.image, 
                model_path=args.model,
                save_results=not args.no_save
            )

if __name__ == "__main__":
    print("=== Raw VQA Predictions (No Selector) ===")
    main() 