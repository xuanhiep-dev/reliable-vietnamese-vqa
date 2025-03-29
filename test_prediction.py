import os
import torch
from inference.predictor import PredictorModeHandler
import matplotlib.pyplot as plt
from PIL import Image

def test_model_prediction():
    """
    Test script to visualize model predictions on a sample image
    """
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = PredictorModeHandler()
    
    # Path to your trained model checkpoint
    model_path = "checkpoints/model_with_selector.pt/checkpoint-150"  # Change this to your checkpoint path
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = predictor.load_final_model(model_path)
    
    # Path to test image (using the example image provided)
    image_path = "example/example.png"
    
    # Test with different questions
    questions = [
        "Ảnh này có gì?",  # What's in this image?
        "Bức ảnh này được chụp ở đâu?",  # Where was this photo taken?
        "Có bao nhiêu người trong ảnh?",  # How many people are in the image?
        "Màu sắc chủ đạo trong ảnh là gì?",  # What is the main color in the image?
    ]
    
    # Display image
    plt.figure(figsize=(10, 8))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Test Image")
    plt.show()
    
    # Test predictions for each question
    print("\n===== MODEL PREDICTIONS =====")
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        predictor.predict_sample(model, image_path, question)
        print("-" * 50)
    
    print("\nPrediction test completed!")

if __name__ == "__main__":
    test_model_prediction() 