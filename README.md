# Example
Install.
```bash
pip install salesforce-lavis
pip install torchscale underthesea mlflow efficientnet_pytorch
pip install --upgrade transformers
pip install --upgrade timm
```
Training VQA from peers.
```bash
!bash scripts/train_vqa_mulmodels_loop.sh lyp_mode=<True, False> use_selector=<True, False>
```

Training VQA model.
```bash
!bash scripts/train_vqa_model.sh use_selector=<True, False> checkpoint_path=<path>
```