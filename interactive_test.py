import os
import torch
from inference.predictor import PredictorModeHandler
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def interactive_test():
    """
    Interactive script to test model predictions with user-provided questions
    """
    parser = argparse.ArgumentParser(description='Test VQA model with your own questions')
    parser.add_argument('--model_path', type=str, default="checkpoints/model_with_selector.pt/checkpoint-150",
                        help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, default="example/example.png",
                        help='Path to the test image')
    args = parser.parse_args()
    
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = PredictorModeHandler()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = predictor.load_final_model(args.model_path)
    
    # Display the image
    try:
        img = Image.open(args.image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Test Image")
        plt.show()
        
        print(f"\nImage loaded: {args.image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print("\n===== INTERACTIVE VQA TESTING =====")
    print("Type 'exit' to quit the program")
    
    while True:
        question = input("\nEnter your question about the image: ")
        
        if question.lower() == 'exit':
            break
            
        if not question.strip():
            print("Please enter a valid question.")
            continue
            
        try:
            predictor.predict_sample(model, args.image_path, question)
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    print("\nTesting completed!")

def custom_image_test():
    """
    Function to test with a custom image
    """
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = PredictorModeHandler()
    
    # Get model path from user
    model_path = input("Enter the path to your model checkpoint (or press Enter for default): ")
    if not model_path:
        model_path = "checkpoints/model_with_selector.pt/checkpoint-150"
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = predictor.load_final_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get image path from user
    while True:
        image_path = input("\nEnter the path to your test image (or 'exit' to quit): ")
        
        if image_path.lower() == 'exit':
            break
            
        if not image_path.strip():
            print("Please enter a valid image path.")
            continue
        
        # Test if image exists
        try:
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Test Image")
            plt.show()
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
        
        # Ask questions about this image
        print("\n===== QUESTIONS ABOUT THIS IMAGE =====")
        print("Type 'next' for a new image or 'exit' to quit")
        
        while True:
            question = input("\nEnter your question about the image: ")
            
            if question.lower() == 'next':
                break
                
            if question.lower() == 'exit':
                return
                
            if not question.strip():
                print("Please enter a valid question.")
                continue
                
            try:
                predictor.predict_sample(model, image_path, question)
            except Exception as e:
                print(f"Error during prediction: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    print("Select test mode:")
    print("1. Interactive testing with default image")
    print("2. Test with custom images")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        interactive_test()
    elif choice == "2":
        custom_image_test()
    else:
        print("Invalid choice. Exiting.") 