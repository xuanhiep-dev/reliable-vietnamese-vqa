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

        self.ckpt_cfg = self._paths_cfg["checkpoint"]
        self.ckpt_cfg["save_path"] = self.ckpt_cfg["save_path"] or "checkpoint/"
        self.ckpt_cfg["load_path"] = self.ckpt_cfg["load_path"] or "checkpoint/"
        self._base_model_path = f"{self.ckpt_cfg['save_path']}/{self._paths_cfg['base_model']}"
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
            torch.save(base_model, self._base_model_path)
            print(f"[INFO] Base model saved to {self._base_model_path}")

    def load_base_model(self):
        if not os.path.exists(self._base_model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self._base_model_path}.")

        print("[INFO] Loading base model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.load(
            self._base_model_path, map_location="cpu", weights_only=False).to(device)
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
        if self._cfg.get("training")["lyp_mode"]:
            self._model = None
            gc.collect()
            self.delete_model()
            torch.cuda.empty_cache()

    # ====== optimizer ======
    def build_optimizer(self):
        if self._use_selector:
            return torch.optim.AdamW(self._model.get_optimizer_parameters())
        return torch.optim.AdamW(self._model.parameters())

    # ====== evaluation ======
    def build_compute_metrics(self):
        def compute_metrics(p):
            logits, labels = p
            preds = np.argmax(logits, axis=1)

            valid_mask = labels != self._unk_index
            labels = labels[valid_mask]
            preds = preds[valid_mask]

            if len(labels) == 0:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                } if self._use_selector else {"accuracy": 0.0}

            if not self._use_selector:
                acc = accuracy_score(labels, preds)
                return {"accuracy": acc}
            else:
                labels = (preds == labels).astype(int)

                acc = accuracy_score(labels, preds)
                prec, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average='binary', zero_division=0
                )
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

        os.makedirs("predictions", exist_ok=True)

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
