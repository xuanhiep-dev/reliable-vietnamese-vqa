import os
import torch
import numpy as np
import json
from inference.predictor import PredictorModeHandler
from utils.dataset import get_sample
import traceback
from PIL import Image
import matplotlib.pyplot as plt

class VQALogitsExtractor(PredictorModeHandler):
    """Extracts raw VQA logits by interfacing with the model's internal structure"""
    
    def __init__(self):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vqa_logits = None
        self.hook_handle = None
    
    def add_vqa_hook(self, model):
        """Add a hook to capture VQA logits before selector"""
        # Unwrap model if it's in DataParallel
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # Register a hook to capture VQA logits
        def hook_fn(module, input, output):
            # Store the VQA logits for later use
            self.vqa_logits = output
            print(f"Captured VQA logits with shape: {output.shape}")
            
        # Try to find where to hook based on model architecture
        if hasattr(actual_model, 'head'):
            print("Adding hook to model's head layer")
            self.hook_handle = actual_model.head.register_forward_hook(hook_fn)
        else:
            print("Could not find appropriate layer to hook")
        
        return model
    
    def remove_hook(self):
        """Remove the hook if it exists"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("Hook removed")
    
    def extract_vqa_logits(self, model, image_path, question, top_k=5, visualize=False):
        """Extract VQA logits directly using hooks"""
        # Reset stored logits
        self.vqa_logits = None
        
        # Get sample
        sample = get_sample(self._paths_cfg, image_path, question)
        
        # Process inputs
        image = sample["image"].to(self._device, dtype=torch.float32)
        question_ids = sample["question"].to(self._device)
        mask = sample["padding_mask"].to(self._device)
        vocab = sample["vocab"]
        idx_to_answer = {v: k for k, v in vocab.items()}
        
        # Run inference
        model.eval()
        with torch.no_grad():
            try:
                # Forward pass through the model
                _ = model(image=image, question=question_ids, padding_mask=mask)
                
                # Check if we captured the VQA logits
                if self.vqa_logits is None:
                    print("Failed to capture VQA logits through hook")
                    return {"status": "error", "message": "Failed to capture VQA logits"}
                
                # Process the captured logits
                logits = self.vqa_logits
                
                # If logits are multi-dimensional, select the first sample
                if len(logits.shape) > 1:
                    logits = logits[0]
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get top-k predictions
                k = min(top_k, probs.size(0))
                if k > 0:
                    top_values, top_indices = torch.topk(probs, k)
                    
                    # Prepare predictions
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
                        print(f"{i+1}. {answer}: {confidence:.4f}")
                    
                    result = {
                        "question": question,
                        "top_prediction": predictions[0],
                        "all_predictions": predictions,
                        "status": "success"
                    }
                    
                    # Visualize if requested
                    if visualize:
                        self._visualize_result(image_path, question, predictions)
                    
                    return result
                else:
                    return {
                        "question": question,
                        "status": "no_predictions",
                        "message": "No predictions available"
                    }
                
            except Exception as e:
                print(f"Error during inference: {e}")
                traceback.print_exc()
                return {
                    "question": question,
                    "status": "error",
                    "message": str(e)
                }
    
    def _visualize_result(self, image_path, question, predictions):
        """Visualize the image with question and top predictions"""
        try:
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # Format title with question and top 3 predictions
            title = f"Q: {question}\n"
            for i, pred in enumerate(predictions[:3]):
                title += f"{i+1}. {pred['answer']} ({pred['confidence']:.4f})"
                if i < min(2, len(predictions)-1):
                    title += "  |  "
            
            plt.title(title)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {e}")

def load_answer_map(answer_file="answers.json"):
    """Load answer mapping from file"""
    try:
        with open(answer_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading answer map: {e}")
        return {}

def process_questions(questions, image_path=None, model_path=None, save_results=True, visualize=False):
    """Process a list of questions and return/save results"""
    # Initialize extractor
    extractor = VQALogitsExtractor()
    
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
    model = extractor.load_final_model(model_path)
    
    # Add hook to the model
    model = extractor.add_vqa_hook(model)
    
    # Load answer map if available
    answer_map = load_answer_map()
    if answer_map:
        print(f"Loaded {len(answer_map)} answers from mapping file")
    
    # Process each question
    results = []
    try:
        for question in questions:
            print(f"\nProcessing question: {question}")
            result = extractor.extract_vqa_logits(model, image_path, question, visualize=visualize)
            results.append(result)
            
            # Show top prediction
            if result["status"] == "success":
                top_pred = result["top_prediction"]
                print(f"Answer: {top_pred['answer']} ({top_pred['confidence']:.4f})")
                
                # Map to canonical answer if available
                if answer_map and top_pred["answer"] in answer_map:
                    canonical = answer_map[top_pred["answer"]]
                    print(f"Canonical answer: {canonical}")
    finally:
        # Clean up hook
        extractor.remove_hook()
    
    # Save results to file if requested
    if save_results:
        output_file = "vqa_logits_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return {"status": "success", "results": results}

def run_interactive_mode(extractor, model, image_path):
    """Run in interactive mode to ask questions"""
    print("\n=== Interactive Mode ===")
    print("Enter your questions (or 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        result = extractor.extract_vqa_logits(model, image_path, question, visualize=True)
        
        if result["status"] == "success":
            print("\nTop 5 Predictions:")
            for pred in result["all_predictions"]:
                print(f"{pred['rank']}. {pred['answer']}: {pred['confidence']:.4f}")

# Function for direct use in Jupyter notebooks
def jupyter_extract_vqa(questions=None, image_path=None, model_path=None, visualize=True):
    """
    Extract VQA predictions in Jupyter notebooks.
    
    Args:
        questions: List of questions or single question string
        image_path: Path to image file
        model_path: Path to model checkpoint
        visualize: Whether to show visualizations
    
    Returns:
        Results dictionary
    """
    print("=== VQA Logits Extractor (Jupyter Mode) ===")
    
    # Handle single question as string
    if isinstance(questions, str):
        questions = [questions]
    
    # Use default questions if none provided
    if not questions:
        questions = [
            "Ảnh này có gì?",  # What's in this image?
            "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
            "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
            "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
        ]
    
    # Process questions
    return process_questions(
        questions, 
        image_path=image_path, 
        model_path=model_path, 
        save_results=True,
        visualize=visualize
    )

def main():
    import argparse
    import sys
    
    # Check if running in Jupyter
    is_jupyter = False
    try:
        # IPython specific check
        if 'ipykernel' in sys.modules:
            is_jupyter = True
            print("Running in Jupyter environment. Use jupyter_extract_vqa() function directly.")
            return
    except:
        pass
    
    # Only run argument parsing outside of Jupyter
    if not is_jupyter:
        parser = argparse.ArgumentParser(description="Extract VQA logits directly from model")
        parser.add_argument("--image", type=str, help="Path to image file")
        parser.add_argument("--questions", type=str, nargs="+", help="Questions to ask")
        parser.add_argument("--model", type=str, help="Path to model checkpoint")
        parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
        parser.add_argument("--no-viz", action="store_true", help="Don't visualize results")
        parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
        
        args = parser.parse_args()
        
        # Initialize extractor
        extractor = VQALogitsExtractor()
        
        # Find model path
        model_path = args.model
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
        
        # Default image path
        image_path = args.image
        if image_path is None:
            image_path = "example/example.png"
            if not os.path.exists(image_path):
                print(f"Default image not found at {image_path}")
                return
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = extractor.load_final_model(model_path)
        
        # Add hook to the model
        model = extractor.add_vqa_hook(model)
        
        try:
            # Interactive mode
            if args.interactive:
                run_interactive_mode(extractor, model, image_path)
            else:
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
                process_questions(
                    questions, 
                    image_path=image_path, 
                    model_path=model_path, 
                    save_results=not args.no_save,
                    visualize=not args.no_viz
                )
        finally:
            # Clean up hook
            extractor.remove_hook()

if __name__ == "__main__":
    main() 