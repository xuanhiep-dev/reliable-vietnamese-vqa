import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from timm.models import create_model
import argparse
from PIL import Image
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Now import modules from the project
from utils.dataset import get_dataset
try:
    from modules.OFA import convert_base_model_to_selective
    print("Using the real OFA module")
except ImportError as e:
    print(f"Warning: Could not import the real OFA module: {e}")
    print("Using a dummy OFA module for testing")
    from scripts.dummy_ofa import convert_base_model_to_selective
from scoring.risk_coverage import RiskCoverage

# Try to import the real model, but use dummy model if it fails
try:
    from modules.model import vivqa_model
    print("Using the real ViVQA model")
except ImportError as e:
    print(f"Warning: Could not import the real model: {e}")
    print("Using a dummy ViVQA model for testing")
    from scripts.dummy_model import dummy_vivqa_model as vivqa_model


def visualize_confidence_distribution(confidences, correctness, save_path="confidence_distribution.png"):
    """Visualize the distribution of confidence scores for correct and incorrect predictions."""
    # Convert to numpy arrays
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(confidences[correctness == 1], bins=20, alpha=0.7, label="Correct Predictions", color="green")
    plt.hist(confidences[correctness == 0], bins=20, alpha=0.7, label="Incorrect Predictions", color="red")
    
    # Add details
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("Distribution of Confidence Scores")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confidence distribution visualization to {save_path}")


def visualize_risk_coverage_curve(confidences, correctness, save_path="risk_coverage.png"):
    """Generate a risk-coverage curve for more detailed visualization."""
    # Convert to numpy arrays
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Sort by confidence (high to low)
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_conf = confidences[sorted_indices]
    sorted_correctness = correctness[sorted_indices]
    
    # Calculate cumulative accuracy and coverage
    n = len(sorted_conf)
    num_elements = np.arange(1, n+1)
    coverage = num_elements / n
    cumulative_risk = (1 - sorted_correctness).cumsum() / num_elements
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot risk-coverage curve
    plt.plot(coverage, cumulative_risk, "b-", linewidth=2)
    
    # Add lines for risk tolerance levels
    risk_levels = [0.01, 0.05, 0.1, 0.2]
    colors = ['green', 'orange', 'red', 'purple']
    
    for risk, color in zip(risk_levels, colors):
        # Find the highest coverage at this risk level
        idx = np.searchsorted(cumulative_risk, risk, side='right') - 1
        if idx >= 0:
            max_coverage = coverage[idx]
            plt.axhline(y=risk, color=color, linestyle=':', alpha=0.7)
            plt.axvline(x=max_coverage, color=color, linestyle=':', alpha=0.7)
            plt.plot([0, max_coverage], [risk, risk], color=color, linewidth=2)
            plt.plot([max_coverage, max_coverage], [0, risk], color=color, linewidth=2)
            plt.annotate(f'C@{risk}={max_coverage:.3f}', 
                        xy=(max_coverage, risk),
                        xytext=(max_coverage + 0.05, risk + 0.01),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10)
    
    # Add details
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Risk-Coverage Curve")
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, min(1, max(cumulative_risk) * 1.1))
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    print(f"Saved risk-coverage curve to {save_path}")


def evaluate_with_risk_coverage(confidences, correctness, results_dir):
    """
    Evaluate the model using the RiskCoverage class instead of fixed thresholds.
    
    Args:
        confidences: NumPy array of confidence scores
        correctness: NumPy array of correctness indicators (1 for correct, 0 for incorrect)
        results_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize RiskCoverage evaluator
    risk_cov_evaluator = RiskCoverage(gather_dist=False)
    
    # Add confidence scores and correctness indicators
    risk_cov_evaluator.add(confidences, correctness)
    
    # Compute metrics
    metrics = risk_cov_evaluator.compute()
    
    # Print metrics
    print("\nRisk-Coverage Evaluation Metrics:")
    print(f"AUC: {metrics['auc']:.4f}")
    
    for risk_level in [0.01, 0.05, 0.1, 0.2]:
        print(f"Coverage at {risk_level*100}% risk (C@{risk_level}): {metrics[f'cov@{risk_level}']:.4f}")
        print(f"Confidence threshold at {risk_level*100}% risk: {metrics[f'thresh@{risk_level}']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(results_dir, "risk_coverage_metrics.txt"), "w") as f:
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        for risk_level in [0.01, 0.05, 0.1, 0.2]:
            f.write(f"Coverage at {risk_level*100}% risk (C@{risk_level}): {metrics[f'cov@{risk_level}']:.4f}\n")
            f.write(f"Confidence threshold at {risk_level*100}% risk: {metrics[f'thresh@{risk_level}']:.4f}\n")
    
    return metrics


def test_selective_vqa():
    parser = argparse.ArgumentParser(description="Test the selective VQA model")
    parser.add_argument("--data-dir", type=str, default="./dummy_data", help="Directory with the dummy dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./vqa_checkpoints", help="Directory to save model and results")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for abstention")
    args = parser.parse_args()
    
    # Directories
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Paths
    # image_path = os.path.join(data_dir, "images")
    # ans_path = os.path.join(data_dir, "vocab.json")
    # train_path = os.path.join(data_dir, "ViVQA-csv", "train.csv")
    # val_path = os.path.join(data_dir, "ViVQA-csv", "val.csv")
    # test_path = os.path.join(data_dir, "ViVQA-csv", "test.csv")
    # base_model_path = os.path.join(checkpoint_dir, "base_model.pth")
    # selective_model_path = os.path.join(checkpoint_dir, "selective_model.pth")
    
    image_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/images"
    ans_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/vocab.json"
    train_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/train.csv"
    val_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/val.csv"
    test_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/test.csv"
    base_model_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/vqa_checkpoints/base_model.pth"
    selective_model_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/vqa_checkpoints/selective_model.pth"
    
    # Prepare data arguments
    class DataArgs:
        def __init__(self):
            self.image_path = image_path
            self.ans_path = ans_path
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
    
    opt = DataArgs()
    
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_dataset(opt)
    
    # Create directory to save results
    results_dir = os.path.join(checkpoint_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create base model if it doesn't exist
    if not os.path.exists(base_model_path):
        print("Creating base model...")
        base_model = vivqa_model(
                      num_classes=len(train_dataset.vocab_a),
                      drop_path_rate=0.3,
                      encoder_layers=2,  # Smaller model for testing
                      encoder_attention_heads=2)  # Smaller model for testing
        
        # Save base model
        torch.save(base_model, base_model_path)
        print(f"Base model saved to {base_model_path}")
    else:
        print(f"Loading base model from {base_model_path}")
        base_model = torch.load(base_model_path, map_location="cpu")
    
    # Convert to selective model
    print("Creating selective model...")
    selective_model = convert_base_model_to_selective(
        base_model=base_model,
        hidden_size=768,
        selective_hidden_1=768,
        selective_hidden_2=768,
        selective_dropout=0.1,
        selective_features="pooled_text+pooled_img+prob+cls_rep",
        freeze_base_model=True,
    )
    
    # Save selective model (untrained)
    torch.save(selective_model, selective_model_path)
    print(f"Selective model saved to {selective_model_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selective_model.to(device)
    selective_model.eval()
    
    # Test the model on a few examples
    print("\nTesting the model on a few examples:")
    
    results = []
    confidence_scores = []
    correctness_values = []
    
    # Get random samples
    import random
    test_indices = random.sample(range(len(test_dataset)), min(5, 10))
    
    for idx in test_indices:
        # Get sample
        sample = test_dataset[idx]
        
        # Move tensors to device
        image = sample["image"].unsqueeze(0).to(device)
        question = sample["question"].unsqueeze(0).to(device)
        padding_mask = sample["padding_mask"].unsqueeze(0).to(device)
        label = sample["labels"]
        
        # Get ground truth answer
        answer_idx = label if isinstance(label, int) else label.item()
        answer_text = list(test_dataset.vocab_a.keys())[list(test_dataset.vocab_a.values()).index(answer_idx)]
        
        # Process with base model first to simulate having a trained model
        base_outputs = base_model(image, question, padding_mask)
        base_probs = F.softmax(base_outputs.logits, dim=-1)
        base_pred_idx = torch.argmax(base_probs, dim=1).item()
        base_pred_text = list(test_dataset.vocab_a.keys())[list(test_dataset.vocab_a.values()).index(base_pred_idx)]
        
        # Because we don't have a trained model, we'll simulate confidence
        # In a real scenario, this would come from the trained selective model
        simulated_confidence = random.random()  # Random confidence between 0 and 1
        
        # Check if the prediction is correct
        is_correct = (base_pred_idx == answer_idx)
        
        # Make the confidence more realistic (higher for correct answers, lower for incorrect)
        if is_correct:
            simulated_confidence = 0.5 + random.random() * 0.5  # Between 0.5 and 1.0
        else:
            simulated_confidence = random.random() * 0.7  # Between 0.0 and 0.7
        
        # Store results
        result = {
            "question": sample["question"].tolist(),
            "prediction_idx": base_pred_idx,
            "prediction_text": base_pred_text,
            "ground_truth_idx": answer_idx,
            "ground_truth_text": answer_text,
            "confidence": simulated_confidence,
            "is_correct": is_correct
        }
        
        # Get sample metadata
        if hasattr(test_dataset, "get_sample_metadata"):
            metadata = test_dataset.get_sample_metadata(idx)
            if metadata:
                result.update(metadata)
        
        results.append(result)
        confidence_scores.append(simulated_confidence)
        correctness_values.append(1 if is_correct else 0)
        
        # Print result
        print(f"\nQuestion: {result.get('question', 'N/A')}")
        print(f"Image ID: {result.get('img_id', 'N/A')}")
        print(f"Ground Truth: {result['ground_truth_text']}")
        print(f"Prediction: {result['prediction_text']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Correct: {'Yes' if result['is_correct'] else 'No'}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)
    print(f"\nSaved test results to {os.path.join(results_dir, 'test_results.csv')}")
    
    # Convert to numpy arrays for evaluation
    confidence_scores = np.array(confidence_scores)
    correctness_values = np.array(correctness_values)
    
    # Evaluate using RiskCoverage class
    print("\nEvaluating using Risk-Coverage metrics:")
    risk_cov_metrics = evaluate_with_risk_coverage(
        confidence_scores, 
        correctness_values,
        results_dir
    )
    
    # Also perform traditional evaluation for comparison
    accuracy_all = correctness_values.mean()
    
    # For each risk threshold, determine the corresponding confidence threshold
    # and calculate metrics using that threshold
    print("\nTraditional Metrics (for comparison):")
    print(f"Overall Accuracy: {accuracy_all:.4f}")
    
    for risk_level in [0.01, 0.05, 0.1, 0.2]:
        conf_threshold = risk_cov_metrics[f'thresh@{risk_level}']
        answered = confidence_scores >= conf_threshold
        
        if answered.sum() > 0:
            accuracy_answered = correctness_values[answered].mean()
            risk = 1.0 - accuracy_answered
            coverage = answered.mean()
            
            print(f"At risk tolerance {risk_level*100}% (threshold={conf_threshold:.4f}):")
            print(f"  Coverage: {coverage:.4f}")
            print(f"  Accuracy on answered: {accuracy_answered:.4f}")
            print(f"  Risk: {risk:.4f}")
    
    # Visualize confidence distribution
    visualize_confidence_distribution(
        confidence_scores, 
        correctness_values,
        save_path=os.path.join(results_dir, "confidence_distribution.png")
    )
    
    # Visualize risk-coverage curve
    visualize_risk_coverage_curve(
        confidence_scores, 
        correctness_values,
        save_path=os.path.join(results_dir, "risk_coverage.png")
    )
    
    print("\nTest complete. Results and visualizations saved to the test_results directory.")


if __name__ == "__main__":
    test_selective_vqa() 