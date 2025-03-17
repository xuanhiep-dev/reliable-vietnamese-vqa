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
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples to evaluate")
    parser.add_argument("--skip-model-creation", action="store_true", help="Skip creating models if they already exist")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick test with minimal evaluation")
    args = parser.parse_args()
    
    # Directories
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    
    # Ensure checkpoint directory exists (preventing the "parent directory doesn't exist" error)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create results directory
    results_dir = os.path.join(checkpoint_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Paths
    image_path = os.path.join(data_dir, "images")
    ans_path = os.path.join(data_dir, "vocab.json")
    train_path = os.path.join(data_dir, "ViVQA-csv", "train.csv")
    val_path = os.path.join(data_dir, "ViVQA-csv", "val.csv")
    test_path = os.path.join(data_dir, "ViVQA-csv", "test.csv")
    base_model_path = os.path.join(checkpoint_dir, "base_model.pth")
    selective_model_path = os.path.join(checkpoint_dir, "selective_model.pth")
    
    # Create model checkpoints directory (if it's a subdirectory)
    model_dir = os.path.dirname(base_model_path)
    if model_dir and model_dir != checkpoint_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Uncomment these lines to use absolute paths for debugging
    # image_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/images"
    # ans_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/vocab.json"
    # train_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/train.csv"
    # val_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/val.csv"
    # test_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/dummy_data/ViVQA-csv/test.csv"
    # base_model_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/vqa_checkpoints/base_model.pth"
    # selective_model_path = "E:/Reliable VQA/Vietnamese-Reliable-VQA/vqa_checkpoints/selective_model.pth"
    
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
    
    # Create base model if it doesn't exist
    if not os.path.exists(base_model_path) and not args.skip_model_creation:
        print("Creating minimal base model for testing selective functionality...")
        base_model = vivqa_model(
                      num_classes=len(train_dataset.vocab_a),
                      drop_path_rate=0.0,  # Disable droppath for testing
                      encoder_layers=1,    # Minimal number of layers
                      encoder_attention_heads=1,  # Minimal attention heads
                      encoder_ffn_embed_dim=512)  # Smaller FFN dimensions
        
        # Save base model
        torch.save(base_model, base_model_path)
        print(f"Minimal base model saved to {base_model_path}")
    else:
        if os.path.exists(base_model_path):
            print(f"Loading base model from {base_model_path}")
            try:
                base_model = torch.load(base_model_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading base model: {e}")
                if args.skip_model_creation:
                    print("Cannot proceed without a valid model. Please remove --skip-model-creation flag.")
                    return
                print("Creating a new base model instead...")
                base_model = vivqa_model(
                          num_classes=len(train_dataset.vocab_a),
                          drop_path_rate=0.0,
                          encoder_layers=1,
                          encoder_attention_heads=1,
                          encoder_ffn_embed_dim=512)
        else:
            if args.skip_model_creation:
                print(f"Base model file not found at {base_model_path} and --skip-model-creation is set.")
                print("Cannot proceed without a valid model. Please remove the flag or provide a valid model.")
                return
            print(f"Base model file not found. Creating a new base model...")
            base_model = vivqa_model(
                      num_classes=len(train_dataset.vocab_a),
                      drop_path_rate=0.0,
                      encoder_layers=1,
                      encoder_attention_heads=1,
                      encoder_ffn_embed_dim=512)
    
    # Convert to selective model if it doesn't exist
    if not os.path.exists(selective_model_path) and not args.skip_model_creation:
        print("Creating selective model with minimal architecture...")
        selective_model = convert_base_model_to_selective(
            base_model=base_model,
            hidden_size=768,
            selective_hidden_1=256,  # Reduced hidden size
            selective_hidden_2=128,  # Reduced hidden size
            selective_dropout=0.0,   # Disable dropout for testing
            selective_features="pooled_text+pooled_img+prob",  # Simplified feature set
            freeze_base_model=True,
        )
        
        # Save selective model
        torch.save(selective_model, selective_model_path)
        print(f"Selective model saved to {selective_model_path}")
    else:
        if os.path.exists(selective_model_path):
            print(f"Loading selective model from {selective_model_path}")
            try:
                selective_model = torch.load(selective_model_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading selective model: {e}")
                if args.skip_model_creation:
                    print("Cannot proceed without a valid model. Please remove --skip-model-creation flag.")
                    return
                print("Creating a new selective model instead...")
                selective_model = convert_base_model_to_selective(
                    base_model=base_model,
                    hidden_size=768,
                    selective_hidden_1=256,
                    selective_hidden_2=128,
                    selective_dropout=0.0,
                    selective_features="pooled_text+pooled_img+prob",
                    freeze_base_model=True,
                )
        else:
            if args.skip_model_creation:
                print(f"Selective model file not found at {selective_model_path} and --skip-model-creation is set.")
                print("Cannot proceed without a valid model. Please remove the flag or provide a valid model.")
                return
            print(f"Selective model file not found. Creating a new selective model...")
            selective_model = convert_base_model_to_selective(
                base_model=base_model,
                hidden_size=768,
                selective_hidden_1=256,
                selective_hidden_2=128,
                selective_dropout=0.0,
                selective_features="pooled_text+pooled_img+prob",
                freeze_base_model=True,
            )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selective_model.to(device)
    selective_model.eval()
    
    # Test the model on samples
    print("\nTesting selective predictions on samples:")
    
    results = []
    confidence_scores = []
    correctness_values = []
    
    # Get samples for testing
    import random
    num_samples = min(len(test_dataset), args.num_samples)
    test_indices = random.sample(range(len(test_dataset)), num_samples)
    
    print(f"Running tests on {num_samples} random samples...")
    
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
        
        # Process with the selective model - get both prediction and confidence
        with torch.no_grad():
            # Make prediction using the selective model
            selective_outputs = selective_model(image, question, padding_mask)
            
            # Get logits and confidence
            selective_logits = selective_outputs.logits
            selective_probs = F.softmax(selective_logits, dim=-1)
            selective_pred_idx = torch.argmax(selective_probs, dim=1).item()
            selective_pred_text = list(test_dataset.vocab_a.keys())[list(test_dataset.vocab_a.values()).index(selective_pred_idx)]
            
            # Get confidence score directly from the model
            model_confidence = selective_outputs.confidence
            # Handle different possible shapes of confidence output
            if hasattr(model_confidence, 'shape') and len(model_confidence.shape) > 0:
                if model_confidence.shape[0] > 1:
                    model_confidence = model_confidence.squeeze()
                model_confidence = model_confidence.item()
        
        # Check if the prediction is correct
        is_correct = (selective_pred_idx == answer_idx)
        
        # Store results
        result = {
            "question": sample["question_text"] if hasattr(sample, "question_text") else "Question text not available",
            "prediction_idx": selective_pred_idx,
            "prediction_text": selective_pred_text,
            "ground_truth_idx": answer_idx,
            "ground_truth_text": answer_text,
            "confidence": model_confidence,
            "is_correct": is_correct
        }
        
        # Try to get the original question text if available
        if hasattr(test_dataset, "metadata") and hasattr(test_dataset.metadata, "question"):
            sample_idx = test_dataset.metadata.index[idx]
            result["question"] = test_dataset.metadata.question[sample_idx]
        elif hasattr(test_dataset, "df") and "question" in test_dataset.df.columns:
            result["question"] = test_dataset.df.iloc[idx]["question"]
        
        # Get sample metadata
        if hasattr(test_dataset, "get_sample_metadata"):
            metadata = test_dataset.get_sample_metadata(idx)
            if metadata:
                result.update(metadata)
        
        results.append(result)
        confidence_scores.append(model_confidence)
        correctness_values.append(1 if is_correct else 0)
        
        # Print result
        print(f"\nQuestion: {result.get('question', 'N/A')}")
        print(f"Image ID: {result.get('img_id', 'N/A')}")
        print(f"Ground Truth: {result['ground_truth_text']}")
        print(f"Prediction: {result['prediction_text']}")
        print(f"Confidence (from model): {result['confidence']:.4f}")
        print(f"Correct: {'Yes' if result['is_correct'] else 'No'}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)
    print(f"\nSaved test results to {os.path.join(results_dir, 'test_results.csv')}")
    
    # Convert to numpy arrays for evaluation
    confidence_scores = np.array(confidence_scores)
    correctness_values = np.array(correctness_values)
    
    # Quick test mode - only show basic results
    if args.quick_test:
        print("\n" + "="*50)
        print("QUICK TEST RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {correctness_values.mean():.4f}")
        
        # Show confidence distribution statistics
        print(f"Average Confidence: {confidence_scores.mean():.4f}")
        print(f"Confidence for Correct Predictions: {confidence_scores[correctness_values == 1].mean():.4f}")
        print(f"Confidence for Incorrect Predictions: {confidence_scores[correctness_values == 0].mean():.4f}")
        
        # Basic threshold evaluation
        threshold = args.threshold
        answered = confidence_scores >= threshold
        if answered.sum() > 0:
            accuracy_answered = correctness_values[answered].mean()
            coverage = answered.mean()
            print(f"\nAt confidence threshold {threshold:.2f}:")
            print(f"  Coverage: {coverage:.4f} ({answered.sum()}/{len(correctness_values)} samples)")
            print(f"  Accuracy on answered: {accuracy_answered:.4f}")
            print(f"  Abstention rate: {1-coverage:.4f}")
        
        print("\nQuick test complete.")
        return
    
    # Full evaluation - proceed with detailed metrics
    print("\nEvaluating using Risk-Coverage metrics:")
    risk_cov_metrics = evaluate_with_risk_coverage(
        confidence_scores, 
        correctness_values,
        results_dir
    )
    
    # Add a new summary evaluation of the selective model
    print("\n" + "="*50)
    print("SELECTIVE MODEL EVALUATION SUMMARY")
    print("="*50)
    
    # Overall accuracy
    accuracy_all = correctness_values.mean()
    print(f"Overall Accuracy: {accuracy_all:.4f}")
    
    # AUC score
    if len(np.unique(correctness_values)) > 1:  # Need both correct and incorrect predictions
        try:
            auc_score = roc_auc_score(correctness_values, confidence_scores)
            print(f"AUC Score: {auc_score:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
    
    # Evaluate at multiple thresholds
    thresholds = [0.5, 0.7, 0.9]
    print("\nPerformance at different confidence thresholds:")
    for threshold in thresholds:
        answered = confidence_scores >= threshold
        if answered.sum() > 0:
            accuracy_answered = correctness_values[answered].mean()
            coverage = answered.mean()
            print(f"\nAt confidence threshold {threshold:.2f}:")
            print(f"  Coverage: {coverage:.4f} ({answered.sum()}/{len(correctness_values)} samples)")
            print(f"  Accuracy on answered: {accuracy_answered:.4f}")
            print(f"  Abstention rate: {1-coverage:.4f}")
        else:
            print(f"\nAt confidence threshold {threshold:.2f}:")
            print("  No samples above this threshold")
    
    # Evaluate at risk-based thresholds
    print("\nPerformance at different risk tolerance levels:")
    for risk_level in [0.01, 0.05, 0.1, 0.2]:
        conf_threshold = risk_cov_metrics[f'thresh@{risk_level}']
        coverage = risk_cov_metrics[f'cov@{risk_level}']
        answered = confidence_scores >= conf_threshold
        
        if answered.sum() > 0:
            accuracy_answered = correctness_values[answered].mean()
            risk = 1.0 - accuracy_answered
            
            print(f"\nAt risk tolerance {risk_level*100}% (threshold={conf_threshold:.4f}):")
            print(f"  Coverage: {coverage:.4f} ({answered.sum()}/{len(correctness_values)} samples)")
            print(f"  Accuracy on answered: {accuracy_answered:.4f}")
            print(f"  Actual risk: {risk:.4f}")
            print(f"  Difference from target risk: {abs(risk-risk_level):.4f}")
        else:
            print(f"\nAt risk tolerance {risk_level*100}% (threshold={conf_threshold:.4f}):")
            print("  No samples above this threshold")
    
    # Save the evaluation summary to a file
    with open(os.path.join(results_dir, "selective_evaluation_summary.txt"), "w") as f:
        f.write("SELECTIVE MODEL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy_all:.4f}\n")
        
        if len(np.unique(correctness_values)) > 1:
            try:
                auc_score = roc_auc_score(correctness_values, confidence_scores)
                f.write(f"AUC Score: {auc_score:.4f}\n")
            except Exception as e:
                f.write(f"Could not calculate AUC: {e}\n")
        
        f.write("\nPerformance at different confidence thresholds:\n")
        for threshold in thresholds:
            answered = confidence_scores >= threshold
            if answered.sum() > 0:
                accuracy_answered = correctness_values[answered].mean()
                coverage = answered.mean()
                f.write(f"\nAt confidence threshold {threshold:.2f}:\n")
                f.write(f"  Coverage: {coverage:.4f} ({answered.sum()}/{len(correctness_values)} samples)\n")
                f.write(f"  Accuracy on answered: {accuracy_answered:.4f}\n")
                f.write(f"  Abstention rate: {1-coverage:.4f}\n")
            else:
                f.write(f"\nAt confidence threshold {threshold:.2f}:\n")
                f.write("  No samples above this threshold\n")
        
        f.write("\nPerformance at different risk tolerance levels:\n")
        for risk_level in [0.01, 0.05, 0.1, 0.2]:
            conf_threshold = risk_cov_metrics[f'thresh@{risk_level}']
            coverage = risk_cov_metrics[f'cov@{risk_level}']
            answered = confidence_scores >= conf_threshold
            
            if answered.sum() > 0:
                accuracy_answered = correctness_values[answered].mean()
                risk = 1.0 - accuracy_answered
                
                f.write(f"\nAt risk tolerance {risk_level*100}% (threshold={conf_threshold:.4f}):\n")
                f.write(f"  Coverage: {coverage:.4f} ({answered.sum()}/{len(correctness_values)} samples)\n")
                f.write(f"  Accuracy on answered: {accuracy_answered:.4f}\n")
                f.write(f"  Actual risk: {risk:.4f}\n")
                f.write(f"  Difference from target risk: {abs(risk-risk_level):.4f}\n")
            else:
                f.write(f"\nAt risk tolerance {risk_level*100}% (threshold={conf_threshold:.4f}):\n")
                f.write("  No samples above this threshold\n")
    
    print(f"\nEvaluation summary saved to {os.path.join(results_dir, 'selective_evaluation_summary.txt')}")
    
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