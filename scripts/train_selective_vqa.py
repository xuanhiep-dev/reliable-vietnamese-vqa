from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import os
import sys
import torch
from torch.nn.functional import softmax
from timm.models import create_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Now import modules from the project
import modules.model
from modules.OFA import convert_base_model_to_selective, SelectiveViVQAOutput
from utils.dataset import get_dataset
from scoring.risk_coverage import RiskCoverage

warnings.filterwarnings("ignore")
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlflow-vivqa-selective'
BASE_MODEL_PATH = "vqa_checkpoints/base_model.pth"
SELECTIVE_MODEL_PATH = "vqa_checkpoints/selective_model.pth"


def compute_metrics(eval_pred):
    """
    Compute metrics for both VQA prediction and selective prediction.
    """
    predictions, labels = eval_pred
    
    # Extract VQA predictions and selective confidence
    vqa_preds = predictions[0]
    confidence_preds = predictions[1]
    
    # Get VQA labels and correctness labels
    vqa_labels = labels[0]
    correctness_labels = labels[1] if len(labels) > 1 else None
    
    # Compute VQA accuracy
    vqa_pred_classes = np.argmax(vqa_preds, axis=1)
    vqa_accuracy = accuracy_score(y_true=vqa_labels, y_pred=vqa_pred_classes)
    
    metrics = {"vqa_accuracy": vqa_accuracy}
    
    # If we have correctness labels, compute AUC for selective prediction
    if correctness_labels is not None:
        # Convert sigmoid output to probabilities
        confidence_probs = 1 / (1 + np.exp(-confidence_preds))
        
        # Compute ROC AUC for confidence prediction
        auc = roc_auc_score(correctness_labels, confidence_probs)
        metrics["selective_auc"] = auc
        
        # Use RiskCoverage for proper risk-coverage metrics
        risk_cov_evaluator = RiskCoverage(gather_dist=False)
        risk_cov_evaluator.add(confidence_probs, correctness_labels)
        risk_cov_metrics = risk_cov_evaluator.compute()
        
        # Add Risk-Coverage metrics
        metrics["risk_coverage_auc"] = risk_cov_metrics["auc"]
        
        # Add coverage at different risk levels
        for risk_level in [0.01, 0.05, 0.1, 0.2]:
            metrics[f"cov@{risk_level}"] = risk_cov_metrics[f"cov@{risk_level}"]
            metrics[f"thresh@{risk_level}"] = risk_cov_metrics[f"thresh@{risk_level}"]
    
    return metrics


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


class SelectiveTrainer(Trainer):
    """
    Custom trainer for selective VQA that handles both VQA and selective prediction.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for both VQA prediction and selective prediction.
        """
        # Extract inputs
        labels = inputs.pop("labels", None)
        
        # Generate correctness labels for selective prediction
        # (1 if the prediction is correct, 0 if incorrect)
        correctness_labels = None
        if labels is not None:
            with torch.no_grad():
                # Get base model predictions
                base_outputs = model.base_model(**inputs)
                pred_classes = torch.argmax(base_outputs.logits, dim=-1)
                correctness_labels = (pred_classes == labels).float()
        
        # Forward pass with selective prediction
        outputs = model(**inputs, labels=labels, confidence_labels=correctness_labels)
        
        # Return loss and outputs
        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Make predictions for evaluation, returning both VQA and selective predictions.
        """
        # Extract inputs and labels
        labels = inputs.pop("labels", None)
        
        # Generate correctness labels for selective prediction
        correctness_labels = None
        if labels is not None:
            with torch.no_grad():
                # Get base model predictions
                base_outputs = model.base_model(**inputs)
                pred_classes = torch.argmax(base_outputs.logits, dim=-1)
                correctness_labels = (pred_classes == labels).float()
        
        # Forward pass with selective prediction
        with torch.no_grad():
            outputs = model(**inputs, labels=labels, confidence_labels=correctness_labels)
        
        # Extract predictions
        vqa_logits = outputs.logits
        confidence_logits = outputs.confidence
        
        if labels is None:
            return None, None, None
        
        # Return predictions and labels
        return (
            outputs.loss.detach(),
            (vqa_logits.detach(), confidence_logits.detach()),
            (labels, correctness_labels)
        )


def get_options():
    args = argparse.ArgumentParser()

    # Base arguments from main.py
    args.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical", "passive"], default="passive")
    args.add_argument("--lr-scheduler-type", choices=["cosine", "linear"], default="cosine")
    args.add_argument("--warmup-ratio", type=float, default=0.1)
    args.add_argument("--logging-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-total-limit", type=int, default=1)
    args.add_argument("-tb", "--train-batch-size", type=int, default=32)
    args.add_argument("-eb", "--eval-batch-size", type=int, default=32)
    args.add_argument("-e", "--epochs", type=int, default=10)
    args.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    args.add_argument("--weight-decay", type=float, default=0.01)
    args.add_argument("--workers", type=int, default=2)

    # Data paths
    args.add_argument("--image-path", type=str, default="./data/images")
    args.add_argument("--ans-path", type=str, default="./data/vocab.json")
    args.add_argument("--train-path", type=str, default="./data/ViVQA-csv/train.csv")
    args.add_argument("--val-path", type=str, default="./data/ViVQA-csv/val.csv")
    args.add_argument("--test-path", type=str, default="./data/ViVQA-csv/test.csv")

    # Model settings
    args.add_argument("--drop-path-rate", type=float, default=0.3)
    args.add_argument("--encoder-layers", type=int, default=6)
    args.add_argument("--encoder-attention-heads-layers", type=int, default=6)
    args.add_argument("--classes", type=int, default=353)
    args.add_argument("--checkpoint-dir", type=str, default="./vqa_checkpoints")
    
    # Selective model settings
    args.add_argument("--selective-features", type=str, default="pooled_text+pooled_img+prob+cls_rep")
    args.add_argument("--selective-hidden-1", type=int, default=768)
    args.add_argument("--selective-hidden-2", type=int, default=768)
    args.add_argument("--selective-dropout", type=float, default=0.1)
    args.add_argument("--confidence-loss-weight", type=float, default=1.0)
    args.add_argument("--freeze-base-model", action="store_true", default=True)

    opt = args.parse_args()
    return opt


def _get_train_config(opt):
    args = TrainingArguments(
        output_dir=f"{opt.checkpoint_dir}/selective_model",
        log_level=opt.log_level,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_ratio=opt.warmup_ratio,
        logging_strategy=opt.logging_strategy,
        save_strategy='epoch',
        save_total_limit=opt.save_total_limit,
        per_device_train_batch_size=opt.train_batch_size,
        per_device_eval_batch_size=opt.eval_batch_size,
        num_train_epochs=opt.epochs,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        dataloader_num_workers=opt.workers,
        report_to='mlflow',
        save_safetensors=False,
        disable_tqdm=False,
        overwrite_output_dir=True,
        metric_for_best_model='cov@0.1',  # Use coverage at 10% risk tolerance as the main metric
        eval_strategy='epoch',
        load_best_model_at_end=True,
        greater_is_better=True
    )
    return args


def save_base_model(opt):
    if not os.path.exists(BASE_MODEL_PATH):
        print("Creating model...")
        base_model = create_model('vivqa_model',
                                num_classes=opt.classes,
                                drop_path_rate=opt.drop_path_rate,
                                encoder_layers=opt.encoder_layers,
                                encoder_attention_heads=opt.encoder_attention_heads_layers)
        torch.save(base_model, BASE_MODEL_PATH)


def load_base_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {BASE_MODEL_PATH}.")

    print("Loading base model...")
    model = torch.load(BASE_MODEL_PATH, map_location="cpu", weights_only=False).to(device)

    return model


def evaluate_with_risk_coverage(confidences, correctness, results_dir):
    """
    Evaluate the model using the RiskCoverage class.
    
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


def train_selective_model():
    """
    Train a selective VQA model on top of the base BEiT3 model.
    """
    opt = get_options()

    # Get datasets
    train_dataset, val_dataset, test_dataset = get_dataset(opt)

    # Load or create base model
    save_base_model(opt)
    base_model = load_base_model()

    # Convert to selective model
    print("Creating selective model...")
    selective_model = convert_base_model_to_selective(
        base_model=base_model,
        hidden_size=768,
        selective_hidden_1=opt.selective_hidden_1,
        selective_hidden_2=opt.selective_hidden_2,
        selective_dropout=opt.selective_dropout,
        selective_features=opt.selective_features,
        freeze_base_model=opt.freeze_base_model,
    )

    # Training configuration
    training_args = _get_train_config(opt)

    # Create trainer
    trainer = SelectiveTrainer(
        model=selective_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train model
    print("Training selective model...")
    trainer.train()

    # Save model
    print(f"Saving model to {SELECTIVE_MODEL_PATH}...")
    torch.save(selective_model, SELECTIVE_MODEL_PATH)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")

    # Create results directory
    results_dir = os.path.join(opt.checkpoint_dir, "selective_results")
    os.makedirs(results_dir, exist_ok=True)

    # Save test results
    with open(os.path.join(results_dir, "test_metrics.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    # Generate predictions with proper risk-coverage evaluation
    print("Generating predictions with risk-coverage evaluation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selective_model.to(device)
    selective_model.eval()

    # Collect predictions and ground truth
    all_predictions = []
    all_confidences = []
    all_is_correct = []
    all_metadata = []

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.eval_batch_size,
        shuffle=False,
        num_workers=opt.workers
    )

    # Process all test samples
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Extract labels
        labels = batch.pop("labels", None)
        
        # Generate predictions
        with torch.no_grad():
            # Get base model predictions first to calculate correctness
            base_outputs = selective_model.base_model(**batch)
            pred_classes = torch.argmax(base_outputs.logits, dim=-1)
            
            # Generate correctness labels
            correctness = (pred_classes == labels).float()
            
            # Get confidence scores from selective model
            outputs = selective_model(
                **batch, 
                labels=labels,
                confidence_labels=correctness
            )
            
            # Convert logits to probabilities 
            confidences = torch.sigmoid(outputs.confidence).cpu().numpy()
            
            # Store predictions and metadata
            for i in range(len(pred_classes)):
                idx = batch_idx * opt.eval_batch_size + i
                if idx < len(test_dataset):
                    pred_idx = pred_classes[i].item()
                    label_idx = labels[i].item()
                    is_correct = (pred_idx == label_idx)
                    confidence = confidences[i]
                    
                    # Try to get metadata if available
                    metadata = {}
                    if hasattr(test_dataset, "get_sample_metadata"):
                        sample_metadata = test_dataset.get_sample_metadata(idx)
                        if sample_metadata:
                            metadata = sample_metadata
                    
                    # Store result
                    all_predictions.append(pred_idx)
                    all_confidences.append(confidence)
                    all_is_correct.append(float(is_correct))
                    all_metadata.append(metadata)

    # Convert to numpy arrays
    all_confidences = np.array(all_confidences)
    all_is_correct = np.array(all_is_correct)
    
    # Perform risk-coverage evaluation
    risk_cov_metrics = evaluate_with_risk_coverage(
        all_confidences,
        all_is_correct,
        results_dir
    )
    
    # Create a dataframe with all prediction results
    results_df = pd.DataFrame({
        "prediction": all_predictions,
        "confidence": all_confidences,
        "is_correct": all_is_correct,
        **{key: [item.get(key, None) for item in all_metadata] for key in all_metadata[0] if all_metadata}
    })
    
    # Save full prediction results
    results_path = os.path.join(results_dir, "test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")
    
    # Generate a report for each risk level
    for risk_level in [0.01, 0.05, 0.1, 0.2]:
        # Get threshold for this risk level
        threshold = risk_cov_metrics[f"thresh@{risk_level}"]
        
        # Apply threshold
        answered = all_confidences >= threshold
        abstained = ~answered
        
        # Calculate metrics
        coverage = answered.mean()
        if answered.sum() > 0:
            accuracy_answered = all_is_correct[answered].mean()
            risk = 1.0 - accuracy_answered
        else:
            accuracy_answered = 0.0
            risk = 0.0
        
        # Print metrics
        print(f"\nAt risk tolerance {risk_level*100}%:")
        print(f"  Confidence threshold: {threshold:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Accuracy on answered questions: {accuracy_answered:.4f}")
        print(f"  Risk: {risk:.4f}")
        print(f"  Questions answered: {answered.sum()} / {len(answered)}")
        print(f"  Questions abstained: {abstained.sum()} / {len(abstained)}")
    
    # Create risk-coverage visualization
    visualize_risk_coverage_curve(
        all_confidences, 
        all_is_correct,
        save_path=os.path.join(results_dir, "risk_coverage_curve.png")
    )


if __name__ == '__main__':
    train_selective_model() 