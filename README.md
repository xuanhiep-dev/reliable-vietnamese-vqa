# Installation & Training Guide
## 1. Install.
```bash
pip install salesforce-lavis
pip install torchscale underthesea mlflow efficientnet_pytorch
pip install --upgrade transformers
pip install --upgrade timm
```
## 2. Training.
1. Train from standard VQA model.
```bash
!bash scripts/train_vqa_model.sh use_selector=<true|false> save_checkpoint_path=<path>
```
2. Train from peers (without using selector)
```bash
!bash scripts/train_vqa_mulmodels_loop.sh subsets=[list]
```
## 3. Inference.
