import os
import torch
import json
from inference.predictor import PredictorModeHandler
import matplotlib.pyplot as plt
from PIL import Image

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
            
            return answer_vocab
        else:
            print("Vocabulary file doesn't contain 'answer' key")
    else:
        print(f"Vocabulary file not found at {vocab_path}")
    
    return None

def analyze_test_dataset():
    """Analyze test dataset to understand distribution of answers"""
    try:
        # Try to load the test CSV
        import pandas as pd
        test_path = "data/full/test.csv"
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            print(f"Test dataset loaded: {len(df)} examples")
            
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

def run_predictor_test():
    """Run basic predictions with the standard predictor"""
    predictor = PredictorModeHandler()
    
    # Try to load the model from checkpoint directory
    model_path = input("\nEnter model path (or press Enter for default): ")
    if not model_path:
        # Try different options for model paths
        options = [
            "checkpoints/model_with_selector.pt",
            "checkpoints/model_with_selector.pt/pytorch_model.bin",
            "checkpoints/model_with_selector.pt/checkpoint-150/pytorch_model.bin"
        ]
        
        for option in options:
            if os.path.exists(option):
                model_path = option
                print(f"Found model at: {model_path}")
                break
        
        if not model_path:
            print("Could not find a valid model checkpoint")
            return
    
    # Load the model
    try:
        print(f"Loading model from {model_path}...")
        model = predictor.load_final_model(model_path)
        
        # Test with example image
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
        
        # Display image first
        plt.figure(figsize=(10, 8))
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title("Test Image")
        plt.axis('off')
        plt.show()
        
        for question in questions:
            print("\n" + "="*50)
            print(f"Question: {question}")
            predictor.predict_sample(model, image_path, question)
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== VQA Model Analysis ===")
    
    # Analyze the vocabulary
    print("\n1. Checking vocabulary...")
    vocab = check_vocabulary()
    
    # Analyze the dataset
    print("\n2. Analyzing test dataset...")
    test_df = analyze_test_dataset()
    
    # Run predictions with standard predictor
    print("\n3. Testing model predictions...")
    run_predictor_test()

if __name__ == "__main__":
    main() 