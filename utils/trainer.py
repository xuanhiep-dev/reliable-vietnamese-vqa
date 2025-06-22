import os
import gc
import shutil
import torch
import pandas as pd
import numpy as np
from utils.config import ConfigLoader
from utils.dataset import Process, ViVQADataset
from timm.models import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TrainingModeHandler:
    def __init__(self, config: ConfigLoader = None):
        self._cfg = config or ConfigLoader.from_cli()
        self._paths_cfg = self._cfg.get("paths")
        self._model_cfg = self._cfg.get("model")
        self._use_selector = self._model_cfg.get("use_selector", False)
        self._unk_index = 0

        self.ckpt_cfg = self._paths_cfg["checkpoints"]
        self.ckpt_cfg["save_path"] = self.ckpt_cfg["save_path"] or "checkpoints/"
        self.ckpt_cfg["load_path"] = self.ckpt_cfg["load_path"] or "checkpoints/"
        self._base_model_path = self.ckpt_cfg["base_model_path"] or os.path.join(
            "checkpoints/base", self._paths_cfg["base_model"])

        self._train_dataset, self._valid_dataset, self._test_dataset = self.get_dataset()
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

    # ====== Pre-processing process ======
    def get_dataset(self, validation=True):
        processor = Process()
        image_path = self._paths_cfg["image_path"]
        ans_path = self._paths_cfg["ans_path"]

        df_train = pd.read_csv(self._paths_cfg["train_path"], index_col=0)
        df_test = pd.read_csv(self._paths_cfg["test_path"], index_col=0)

        train_dataset = ViVQADataset(
            df_train, processor, image_path, ans_path)
        test_dataset = ViVQADataset(
            df_test, processor, image_path, ans_path)

        if validation:
            df_val = pd.read_csv(self._paths_cfg["valid_path"], index_col=0)
            val_dataset = ViVQADataset(
                df_val, processor, image_path, ans_path)
            return train_dataset, val_dataset, test_dataset

        return train_dataset, test_dataset

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
            preds = np.argmax(logits, axis=1)

            if not self._use_selector:
                acc = accuracy_score(labels, preds)
                return {"accuracy": acc}
            else:
                correctness = (preds == labels).astype(int)
                correctness = np.array(correctness, dtype=int)
                preds = np.array(preds, dtype=int)

                acc = accuracy_score(correctness, preds)
                prec, recall, f1, _ = precision_recall_fscore_support(
                    correctness, preds, average='binary', zero_division=0
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
        return {v: k for k, v in self._test_dataset.get_vocab().items()}

    def _predict_answers(self, trainer, index_to_answer):
        predictions = trainer.predict(self._test_dataset)
        logits = torch.tensor(predictions.predictions)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predicted_index = torch.argmax(probabilities, dim=-1).numpy()
        predicted_answer = [index_to_answer[idx] for idx in predicted_index]
        confidence_scores = torch.max(probabilities, dim=-1).values.numpy()
        ground_truth = [self._test_dataset.vocab_a[idx]
                        for idx in self._test_dataset["labels"]]

        return predicted_answer, confidence_scores, ground_truth

    def _save_predictions(self, predicted_answer, confidence_scores, ground_truth):
        questions, img_ids = zip(*self._extract_metadata())

        df = pd.DataFrame({
            "question": questions,
            "img_id": img_ids,
            "answer": np.array(predicted_answer, dtype=str),
            "confidence": np.array(confidence_scores, dtype=np.float32),
            "ground_truth": np.array(ground_truth, dtype=str)
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
