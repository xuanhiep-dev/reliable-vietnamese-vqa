from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
import models.model
from utils.trainer import TrainingModeHandler
import pandas as pd
import numpy as np
import mlflow
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlflow-vivqa'


def _get_train_config(cfg):
    training_cfg = cfg.get("training")
    ckpt_cfg = cfg.get("paths")["checkpoints"]
    ckpt_cfg["save_path"] = ckpt_cfg["save_path"] or "checkpoints/"

    args = TrainingArguments(
        output_dir=ckpt_cfg["save_path"],
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
        run_name=training_cfg["run_name"],
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


def train():
    handler = TrainingModeHandler()

    model = handler.load_model()
    optimizer = handler.build_optimizer()
    compute_metrics = handler.build_compute_metrics()

    args = _get_train_config(handler.config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=handler.train_dataset,
        eval_dataset=handler.valid_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=15), PrintMessageCallback(handler.config)]
    )

    trainer.train()
    handler.post_train(trainer)

    mlflow.end_run()
    handler.cleanup_after_training()


if __name__ == '__main__':
    train()
