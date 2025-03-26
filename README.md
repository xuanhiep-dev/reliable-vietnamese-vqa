# Installation & Training Guide
## 1. Install.
```bash
pip install salesforce-lavis
pip install torchscale underthesea mlflow efficientnet_pytorch
pip install --upgrade transformers
pip install --upgrade timm
```
## 2. Training.
### 2.1. Train from standard VQA model.
```bash
!bash scripts/train_vqa_model.sh use_selector=<true|false> save_checkpoint_path=<path>
```
### 2.2. Train from peers (without using selector)
```bash
!bash scripts/train_vqa_mulmodels_loop.sh subsets=subset_id1,subset_id2,...
```
## 3. Inference.
### 3.1. Load VQA model.
```bash
from inference.predictor import PredictorModeHandler

predictor = PredictorModeHandler()
model_path = "/kaggle/working/new_output/checkpoint-1/pytorch_model.bin"
model = predictor.load_final_model(model_path)
```
### 3.2. Get a result.
```bash
image_path = "data/images/119776.jpg"
question = "Con vật trong ảnh màu gì ?"
predictor.predict_sample(model, image_path, question)
```
![Sample Result](example/example.png)