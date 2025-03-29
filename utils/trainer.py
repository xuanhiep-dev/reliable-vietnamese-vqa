import os
import gc
import shutil
import torch
import pandas as pd
import numpy as np
from utils.config import ConfigLoader
from utils.dataset import get_dataset
from timm.models import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TrainingModeHandler:
    def __init__(self, config: ConfigLoader = None):
        self._cfg = config or ConfigLoader.from_cli()
        self._train_dataset, self._valid_dataset, self._test_dataset = get_dataset(
            self._cfg)
        self._paths_cfg = self._cfg.get("paths")
        self._model_cfg = self._cfg.get("model")
        self._use_selector = self._model_cfg.get("use_selector", False)
        self._unk_index = 0

        self.ckpt_cfg = self._paths_cfg["checkpoints"]
        self.ckpt_cfg["save_path"] = self.ckpt_cfg["save_path"] or "checkpoints/"
        self.ckpt_cfg["load_path"] = self.ckpt_cfg["load_path"] or "checkpoints/"
        self._base_model_path = self.ckpt_cfg["base_model_path"] or os.path.join(
            "checkpoints/base", self._paths_cfg["base_model"])
        self._model = None

    # ====== Properties ======
    @property
    def config(self):
        return self._cfg

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def model(self):
        return self._model

    # ====== Model Operations ======
    def save_base_model(self):
        if not os.path.exists(self._base_model_path):
            checkpoint_dir = os.path.dirname(self._base_model_path)
            os.makedirs(checkpoint_dir, exist_ok=True)

            print("[INFO] Creating base model...")
            base_model = create_model("avivqa_model", **self._model_cfg)
            torch.save(base_model.state_dict(), self._base_model_path)
            print(
                f"[INFO] Base model weights saved to {self._base_model_path}")

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path = self.ckpt_cfg["load_path"]

        model = create_model("avivqa_model", **self._model_cfg)
        if not os.path.exists(load_path):
            print(
                f"[INFO] Checkpoint not found at: {load_path}. Creating and loading base model.")
            self.save_base_model()

            if not os.path.exists(self._base_model_path):
                raise FileNotFoundError(
                    f"Base model not found at {self._base_model_path}.")

            print("[INFO] Loading weights from base model...")
            state_dict = torch.load(
                self._base_model_path, map_location="cpu", weights_only=True)
        else:
            print(f"[INFO] Loading checkpoint from {load_path}...")
            state_dict = torch.load(
                load_path, map_location="cpu", weights_only=True)

        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=False)
        self._model = model.to(device)

        return self._model

    def delete_model(self):
        model_path = os.path.join(
            self.ckpt_cfg["save_path"], f"model-{self._cfg.get('training')['subset_id']}")
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
                print(f"[INFO] Directory {model_path} has been deleted.")
            elif os.path.isfile(model_path):
                os.remove(model_path)
                print(f"[INFO] Model file {model_path} has been deleted.")
        else:
            print(
                f"[WARN] Model file or directory {model_path} does not exist.")

    def cleanup_after_training(self):
        self._model = None
        torch.cuda.empty_cache()
        gc.collect()
        if self._cfg.get("training")["lyp_mode"]:
            self.delete_model()

    # ====== optimizer ======
    def build_optimizer(self):
        if self._use_selector:
            return torch.optim.AdamW(self._model.get_optimizer_parameters())
        return torch.optim.AdamW(self._model.parameters())

    # ====== evaluation ======
    def build_compute_metrics(self):
        def compute_metrics(p):
            logits, labels = p
            
            # Debug information
            print(f"DEBUG: Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            print(f"DEBUG: Logits sample: {logits[0][:5]}, Labels sample: {labels[0:5]}")
            
            # Filter out unknown labels first
            valid_mask = labels != self._unk_index
            filtered_labels = labels[valid_mask]
            filtered_logits = logits[valid_mask]
            
            print(f"DEBUG: After filtering - Logits shape: {filtered_logits.shape}, Labels shape: {filtered_labels.shape}")
            if len(filtered_labels) == 0:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                } if self._use_selector else {"accuracy": 0.0}

            if not self._use_selector:
                # For standard VQA model
                preds = np.argmax(filtered_logits, axis=1)
                acc = accuracy_score(filtered_labels, preds)
                print(f"DEBUG: Standard VQA - Accuracy: {acc}")
                return {"accuracy": acc}
            else:
                # For selector model
                # First get VQA predictions
                # Print shape information for debugging
                print(f"DEBUG: Filtered logits shape for selector: {filtered_logits.shape}")
                
                # Check if we have 2 classes (binary classification)
                if filtered_logits.shape[1] == 2:
                    # Binary classification
                    print("DEBUG: Using binary classification logic")
                    # Selector predictions - probability of being correct (class 1)
                    selector_probs = filtered_logits[:, 1]
                    selector_preds = (selector_probs > 0.5).astype(int)
                    
                    # Get VQA predictions for ground truth correctness
                    vqa_preds = np.argmax(filtered_logits, axis=1)
                    gt_correctness = (vqa_preds == filtered_labels).astype(int)
                    
                    # Count distribution of predictions and ground truth
                    print(f"DEBUG: GT correctness distribution: 0s={np.sum(gt_correctness==0)}, 1s={np.sum(gt_correctness==1)}")
                    print(f"DEBUG: Selector preds distribution: 0s={np.sum(selector_preds==0)}, 1s={np.sum(selector_preds==1)}")
                    
                    # Calculate metrics
                    acc = accuracy_score(gt_correctness, selector_preds)
                    prec, recall, f1, _ = precision_recall_fscore_support(
                        gt_correctness, selector_preds, average='binary', zero_division=0
                    )
                else:
                    # Non-binary case - treat as standard VQA
                    print("DEBUG: Using multi-class logic")
                    vqa_preds = np.argmax(filtered_logits, axis=1)
                    gt_correctness = (vqa_preds == filtered_labels).astype(int)
                    
                    # Just use accuracy for now
                    acc = np.mean(gt_correctness)
                    prec, recall, f1 = 0.0, 0.0, 0.0
                
                print(f"DEBUG: Selector metrics - Acc: {acc}, Prec: {prec}, Rec: {recall}, F1: {f1}")
                return {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": recall,
                    "f1": f1
                }

        return compute_metrics

    # ====== Post-training process ======
    def post_train(self, trainer):
        if not self._use_selector:
            index_to_answer = self._get_index_to_answer()
            predicted_answer, confidence_scores = self._predict_answers(
                trainer, index_to_answer)
            self._save_predictions(predicted_answer, confidence_scores)
            self._evaluate_model(trainer)

    def _evaluate_model(self, trainer):
        results = trainer.evaluate(self._test_dataset)
        accuracy = results.get("eval_accuracy")
        print(f"[RESULT] Test Accuracy: {accuracy:.4f}")

    # ====== Prediction ======
    def _get_index_to_answer(self):
        return {v: k for k, v in self._test_dataset.get_answers().items()}

    def _predict_answers(self, trainer, index_to_answer):
        predictions = trainer.predict(self._test_dataset)
        logits = torch.tensor(predictions.predictions)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predicted_index = torch.argmax(probabilities, dim=-1).numpy()
        predicted_answer = [index_to_answer[idx] for idx in predicted_index]
        confidence_scores = torch.max(probabilities, dim=-1).values.numpy()

        return predicted_answer, confidence_scores

    def _save_predictions(self, predicted_answer, confidence_scores):
        questions, img_ids = zip(*self._extract_metadata())

        df = pd.DataFrame({
            "question": questions,
            "img_id": img_ids,
            "answer": np.array(predicted_answer, dtype=str),
            "confidence": np.array(confidence_scores, dtype=np.float32)
        })

        os.makedirs(self._paths_cfg["prediction_path"], exist_ok=True)
        subset_id = self._cfg.get("training")["subset_id"]
        filename = f"answers-{subset_id}.csv" if subset_id is not None else "answers.csv"
        output_path = os.path.join(
            self._paths_cfg["prediction_path"], filename)

        df.to_csv(output_path, encoding="utf-8")
        print(f"[INFO] Predictions saved to {output_path}")

    def _extract_metadata(self):
        for idx in range(len(self._test_dataset)):
            metadata = self._test_dataset.get_sample_metadata(idx)
            if metadata:
                yield metadata["question"], metadata["img_id"]
