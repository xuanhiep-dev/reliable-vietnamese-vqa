# Reliable Vietnamese VQA

A dependable Visual Question Answering (VQA) system designed specifically for the Vietnamese language. It combines visual and language understanding with a selective answering strategy that avoids responding to questions when the model lacks sufficient confidence, helping improve output credibility in real-world scenarios.

---

## Overview

This system enables **accurate** and **selective** answering of visual questions in Vietnamese by combining:

- **Visual Encoder:** Extracts image features using a frozen BLIP-2 model.
- **Text Encoder/Decoder:** Processes Vietnamese questions using BARTpho.
- **Selector Module:** Determines whether the model should answer or abstain based on confidence.
- **Answer Generator:** Generates answers from visual and textual embeddings if confidence is high enough.

---

## Architecture
```
+------------------+
|  Input Image     |
+--------+---------+
        ↓
[BLIP-2 Visual Encoder]
        ↓
+--------+---------+
| Visual Embedding |
+--------+---------+
        ↓
+----------------+----------------+
| Vietnamese Question (BARTpho) |
+---------------+-----------------+
        ↓
[Cross-modal Fusion]
        ↓
+-----------------------+
| Selector (Confidence) |
+-----------+-----------+
Yes (High) | No (Low)
       ↓ | ↓
[Answer Decoder] | [Return "I don't know"]
         ↓
    Final Output
```
## Project Structure
```
reliable-vietnamese-vqa/
│
├── configs/               # Config files (YAML)
├── data/                  # Dataset, vocab, predictions
│   ├── full/
│   ├── vivqa/
│   ├── subsets/
│   ├── predictions/
│   └── vocab.json
│
├── evaluation/            # Evaluation scripts
├── inference/             # Inference pipeline
├── models/                # Model definitions
├── scripts/               # Training shell scripts
├── utils/                 # Utility functions (dataset, training, vision)
│
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

# Installation & Training Guide
## 1. Install.
```bash
pip install salesforce-lavis
pip install torchscale underthesea mlflow efficientnet_pytorch
pip install --upgrade transformers
pip install --upgrade timm
```
## 2. Training.
### 2.1. Train a new model and save checkpoint.
```bash
!bash scripts/train_vqa_model.sh use_selector=<true|false> save_checkpoint_path=<path>
```
### 2.2. Resume training from existing checkpoint.
```bash
!bash scripts/train_vqa_model.sh use_selector=<true|false> load_checkpoint_path=<path>
```
### 2.3. Train from peers (multi-subset training without selector).
```bash
!bash scripts/train_vqa_mulmodels_loop.sh subsets=subset_id1,subset_id2,...
```
## 3. Evaluation.
### 3.1. Load existing model.
```bash
from inference.predictor import PredictorModeHandler

predictor = PredictorModeHandler()
model = predictor.load_final_model(<path_to_selective_model>)
```
### 3.2. Get the prediction and save result.
```bash
predictor.predict_test_dataset(<loaded_model>, <path_to_test_dataset>)
```
### 3.3. Evaluate the result.
```bash
from evaluation.evaluate import EvaluatorModeHandler
import pandas as pd

df = pd.read_csv(<path_to_model_answers>)
evaluator = EvaluatorModeHandler(df)
evaluator.evaluate()
```
## 4. Inference.
### 4.1. Load existing model.
```bash
from inference.predictor import PredictorModeHandler

predictor = PredictorModeHandler()
model_path = <path_to_existing_model>
model = predictor.load_final_model(model_path)
```
### 4.2. Get the model output.
```bash
image_path = <your_image_path>
question = <your_question>
predictor.predict_sample(model, image_path, question)
```
### 5. Features
- Selective Answering to avoid unreliable outputs
- Fully supports Vietnamese input questions
- Modular design: Easily swap visual or language backbones
- Built-in metrics: Accuracy, F1, Answerability