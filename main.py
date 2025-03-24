from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
import torch
from sklearn.metrics import accuracy_score
import models.model
from timm.models import create_model
from utils.dataset import get_sample, process_punctuation
from utils.config import ConfigLoader
from utils.trainer import TrainingModeHandler
import pandas as pd
import numpy as np
import mlflow
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlflow-vivqa'


def load_config():
    return ConfigLoader()


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


def _get_train_config(cfg):
    training_cfg = cfg.get("training")
    cfg.get("paths")["checkpoint_path"] = cfg.get(
        "paths")["checkpoint_path"] or "checkpoint/"

    args = TrainingArguments(
        output_dir=cfg.get("paths")["checkpoint_path"],
        log_level=training_cfg["log_level"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        logging_strategy=training_cfg["logging_strategy"],
        save_strategy=training_cfg["save_strategy"],
        save_total_limit=training_cfg["save_total_limit"],
        per_device_train_batch_size=training_cfg["train_batch_size"],
        per_device_eval_batch_size=training_cfg["eval_batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        dataloader_num_workers=training_cfg["workers"],
        report_to=training_cfg["report_to"],
        save_safetensors=training_cfg["save_safetensors"],
        disable_tqdm=training_cfg["disable_tqdm"],
        overwrite_output_dir=training_cfg["overwrite_output_dir"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        eval_strategy=training_cfg["eval_strategy"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        greater_is_better=training_cfg["greater_is_better"]
    )
    return args


class PrintMessageCallback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        if self.cfg.get("model")["use_selector"]:
            print("Selector is ON. Computing Selective loss.")
        else:
            print(
                "Selector is OFF. Only getting logits from VQA model and computing VQA loss.")


def load_final_model(model_name="avivqa_model", model_path=None, num_classes=353, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = create_model(model_name, pretrained=False,
                             num_classes=num_classes, **kwargs)
    except Exception as e:
        raise ValueError(f"Cannot create the model `{model_name}`: {e}")

    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print("Checkpoint path does not exist")

    return model.to(device).eval()


def predict_sample(image_path, question, ans_path, model):
    print(
        f"[INFO] Inference on image: {image_path}\n Question: {question}")

    sample = get_sample(image_path, question)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ans_path, 'r') as f:
        vocab = {v: process_punctuation(k.lower())
                 for k, v in json.load(f)["answer"].items()}
    image = sample["image"].to(device, dtype=torch.float32)
    question_ids = sample["question"].to(device)
    mask = sample["padding_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image=image, question=question_ids,
                       padding_mask=mask).logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    answer = vocab.get(pred_idx, "I don't know")

    print(f"[RESULT] Answer: {answer} (Confidence: {confidence:.4f})")
    return answer, confidence


def train():
    handler = TrainingModeHandler()

    handler.save_base_model()
    model = handler.load_base_model()
    optimizer = handler.build_optimizer()

    args = _get_train_config(handler.config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=handler.train_dataset,
        eval_dataset=handler.valid_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5), PrintMessageCallback(handler.config)]
    )

    trainer.train()
    handler.post_train(trainer)

    mlflow.end_run()
    handler.cleanup_after_training()


if __name__ == '__main__':
    train()
