from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from torch.nn.functional import softmax
from timm.models import create_model
import torch
from sklearn.metrics import accuracy_score
import modules.model
from utils.dataset import get_dataset
import pandas as pd
import numpy as np
import mlflow
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlflow-vivqa'
BASE_MODEL_PATH = "vqa_checkpoints/base_model.pth"


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


def get_options():
    args = argparse.ArgumentParser()

    # Training Argument
    args.add_argument("--log-level", choices=[
                      "debug", "info", "warning", "error", "critical", "passive"], default="passive")
    args.add_argument("--lr-scheduler-type",
                      choices=["cosine", "linear"], default="cosine")
    args.add_argument("--warmup-ratio", type=float, default=0.1)
    args.add_argument("--logging-strategy",
                      choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-strategy",
                      choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-total-limit", type=int, default=1)
    args.add_argument("-tb", "--train-batch-size", type=int, default=65)
    args.add_argument("-eb", "--eval-batch-size", type=int, default=65)
    args.add_argument("-e", "--epochs", type=int, default=15)
    args.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    args.add_argument("--weight-decay", type=float, default=0.01)
    args.add_argument("--workers", type=int, default=2)

    # Varriables setting
    args.add_argument("--image-path", type=str, default="./data/images")
    args.add_argument("--ans-path", type=str, default="./data/vocab.json")
    args.add_argument("--train-path", type=str,
                      default="./data/ViVQA-csv/train.csv")
    args.add_argument("--val-path", type=str,
                      default="./data/ViVQA-csv/val.csv")
    args.add_argument("--test-path", type=str,
                      default="./data/ViVQA-csv/test.csv")
    # args.add_argument("--feature-paths", type=str, default="./features")

    # Model setting
    # args.add_argument("--efficientnet-b", choices=[0, 1, 2, 3, 4, 5, 6, 7], default=7)
    args.add_argument("--drop-path-rate", type=float, default=0.3)
    args.add_argument("--encoder-layers", type=int, default=6)
    args.add_argument("--encoder-attention-heads-layers", type=int, default=6)
    args.add_argument("--classes", type=int, default=353)
    args.add_argument("--checkpoint-dir", type=str,
                      default="./vqa_checkpoints")
    args.add_argument("--sub-id", type=int, default=1)
    args.add_argument("--predictions-dir", type=str,
                      default="./data/multi-predictions")

    opt = args.parse_args()
    return opt


def _get_train_all_config(opt):
    args = TrainingArguments(
        output_dir=opt.output_dir,
        log_level=opt.log_level,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_ratio=opt.warmup_ratio,
        logging_strategy=opt.logging_strategy,
        save_strategy=opt.save_strategy,
        save_total_limit=opt.save_total_limit,
        per_device_train_batch_size=opt.train_batch_size,
        per_device_eval_batch_size=opt.eval_batch_size,
        num_train_epochs=opt.epochs,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        dataloader_num_workers=opt.workers,
        report_to='mlflow',
        save_safetensors=False,
        disable_tqdm=False
    )
    return args


def _get_train_config(opt):
    args = TrainingArguments(
        output_dir=f"{opt.checkpoint_dir}/model-{opt.sub_id}",
        log_level=opt.log_level,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_ratio=opt.warmup_ratio,
        logging_strategy=opt.logging_strategy,
        save_strategy=opt.save_strategy,
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
        metric_for_best_model='accuracy',
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
        raise FileNotFoundError(
            f"Model checkpoint not found at {BASE_MODEL_PATH}.")

    print("Loading model...")
    model = torch.load(BASE_MODEL_PATH, map_location="cpu").to(device)

    return model


def extract_metadata(test_dataset):
    for idx in range(len(test_dataset)):
        metadata = test_dataset.get_sample_metadata(idx)
        if metadata:
            yield metadata["question"], metadata["img_id"]


def main():
    opt = get_options()

    train_dataset, val_dataset, test_dataset = get_dataset(opt)

    save_base_model(opt)
    model = load_base_model()

    args = _get_train_config(opt)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    logits = torch.tensor(predictions.predictions)
    probabilities = softmax(logits, dim=-1)

    predicted_labels = torch.argmax(probabilities, dim=-1).numpy()
    confidence_scores = torch.max(probabilities, dim=-1).values.numpy()
    questions, img_ids = zip(*extract_metadata(test_dataset))

    df = pd.DataFrame({
        "question": questions,
        "img_id": img_ids,
        "predicted_answer": np.array(predicted_labels, dtype=str),
        "confidence": np.array(confidence_scores, dtype=np.float32)
    })

    predictions_file = f"{opt.predictions_dir}/predictions-{opt.sub_id}.json"
    df.to_csv(predictions_file, encoding="utf-8-sig")

    test = trainer.evaluate(test_dataset)
    print(f'Test Accuracy: {test["eval_accuracy"]}')

    mlflow.end_run()

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
