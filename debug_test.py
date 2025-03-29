import os
import torch
import numpy as np
from inference.predictor import PredictorModeHandler
import matplotlib.pyplot as plt
from PIL import Image
import json

class DebugPredictor(PredictorModeHandler):
    """Extended predictor with debugging capabilities"""
    
    def predict_sample_debug(self, model, image_path, question):
        """Predict with additional debugging information"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Question: {question}")

        sample = self.get_sample(self._paths_cfg, image_path, question)
        image = sample["image"].to(device, dtype=torch.float32)
        question_ids = sample["question"].to(device)
        mask = sample["padding_mask"].to(device)
        
        # Get the vocabulary for debugging
        vocab = sample["vocab"]
        sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[0])}
        
        model.eval()
        with torch.no_grad():
            # Get model output
            output = model(image=image, question=question_ids, padding_mask=mask)
            
            # Debug the model output
            print(f"[DEBUG] Output type: {type(output)}")
            if hasattr(output, 'logits'):
                logits = output.logits
                print(f"[DEBUG] Logits shape: {logits.shape}")
            else:
                logits = output
                print(f"[DEBUG] Output treated as logits, shape: {logits.shape}")
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            
            # Get top 5 predictions
            top_values, top_indices = torch.topk(probs[0], 5)
            top_answers = []
            for i, idx in enumerate(top_indices):
                idx_val = idx.item()
                answer = vocab.get(idx_val, f"unknown-{idx_val}")
                confidence_val = top_values[i].item()
                top_answers.append((answer, confidence_val))
            
            # Main answer
            answer = vocab.get(pred_idx, f"unknown-{pred_idx}")
            
            # Print results
            print(f"[RESULT] Answer: {answer} (Confidence: {confidence:.4f})")
            print("\nTop 5 predictions:")
            for i, (ans, conf) in enumerate(top_answers):
                print(f"{i+1}. {ans}: {conf:.4f}")
            
            # Display the image
            plt.figure(figsize=(10, 8))
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"Q: {question}\nA: {answer} ({confidence:.4f})")
            plt.axis('off')
            plt.show()
            
            return answer, confidence, top_answers

def analyze_test_dataset():
    """Analyze test dataset to understand distribution of answers"""
    try:
        # Try to load the test CSV
        import pandas as pd
        test_path = "data/full/test.csv"
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            print(f"Test dataset loaded: {len(df)} examples")
            
            # Check for the sample image in the dataset
            image_id = "314710"
            samples = df[df['img_id'] == int(image_id)]
            if not samples.empty:
                print(f"\nFound {len(samples)} examples for image {image_id}:")
                for i, row in samples.iterrows():
                    print(f"Question: {row['question']}")
                    print(f"Answer: {row['answer']}")
                    print("-" * 50)
            else:
                print(f"\nImage {image_id} not found in test dataset")
                
            # Analyze answer distribution
            answer_counts = df['answer'].value_counts()
            print(f"\nTop 10 most common answers:")
            print(answer_counts.head(10))
            
            # Check for 'unk' answers
            if 'unk' in answer_counts:
                print(f"\n'unk' appears {answer_counts['unk']} times ({answer_counts['unk']/len(df)*100:.2f}% of dataset)")
            else:
                print("\n'unk' does not appear in the dataset")
                
            return df
        else:
            print(f"Test dataset not found at {test_path}")
            return None
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

def check_vocabulary():
    """Check the model's vocabulary file"""
    vocab_path = "data/vocab.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        if 'answer' in vocab:
            answer_vocab = vocab['answer']
            print(f"Answer vocabulary loaded: {len(answer_vocab)} entries")
            
            # Check if 'unk' is in vocabulary
            if '<unk>' in answer_vocab:
                print(f"'<unk>' is in vocabulary with index {answer_vocab['<unk>']}")
            elif 'unk' in answer_vocab:
                print(f"'unk' is in vocabulary with index {answer_vocab['unk']}")
            else:
                print("Neither '<unk>' nor 'unk' found in vocabulary")
                
            # Print a few sample entries
            print("\nSample vocabulary entries:")
            sample_items = list(answer_vocab.items())[:10]
            for answer, idx in sample_items:
                print(f"{answer}: {idx}")
        else:
            print("Vocabulary file doesn't contain 'answer' key")
    else:
        print(f"Vocabulary file not found at {vocab_path}")

def main():
    print("=== VQA Model Debugging ===")
    
    # Analyze the dataset and vocabulary
    print("\n1. Analyzing test dataset...")
    test_df = analyze_test_dataset()
    
    print("\n2. Checking vocabulary...")
    check_vocabulary()
    
    # Initialize the debug predictor
    print("\n3. Testing model predictions...")
    predictor = DebugPredictor()
    
    # Define model path - adjust as needed
    model_path = input("\nEnter model path (or press Enter for default): ")
    if not model_path:
        model_path = "checkpoints/model_with_selector.pt" 
        # Just the directory, not specific checkpoint, to see if the predictor handles it
    
    # Load the model
    try:
        print(f"Loading model from {model_path}...")
        model = predictor.load_final_model(model_path)
        
        # Test specific image
        image_id = "314710"
        image_path = f"data/images/{image_id}.jpg"
        
        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}")
            # Try with example image
            image_path = "example/example.png"
            if not os.path.exists(image_path):
                print(f"Example image not found at {image_path}")
                return
            
        # Test with different questions
        questions = [
            "Ảnh này có gì?",  # What's in this image?
            "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
            "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
            "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
        ]
        
        for question in questions:
            print("\n" + "="*50)
            predictor.predict_sample_debug(model, image_path, question)
            
        # Interactive mode
        print("\n" + "="*50)
        print("\nEnter your own questions (type 'exit' to quit):")
        while True:
            question = input("\nQuestion: ")
            if question.lower() == 'exit':
                break
            predictor.predict_sample_debug(model, image_path, question)
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 