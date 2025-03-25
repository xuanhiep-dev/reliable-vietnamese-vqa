import os
import torch
import models.model
from utils.config import ConfigLoader
from timm.models import create_model
from utils.dataset import get_sample
from PIL import Image
import matplotlib.pyplot as plt


class PredictorModeHandler:
    def __init__(self):
        self._cfg = ConfigLoader()
        self._paths_cfg = self._cfg.get("paths")
        self._model_cfg = self._cfg.get("model")

    def load_final_model(self, model_path=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = create_model("avivqa_model", **self._model_cfg)
        except Exception as e:
            raise ValueError(f"Cannot create the model: {e}")

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

    def predict_sample(self, model, image_path, question):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"[INFO] Inference on image: {image_path}\n Question: {question}")

        sample = get_sample(self._paths_cfg, image_path, question)
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

        answer = sample["vocab"].get(pred_idx, "I don't know")

        print(f"[RESULT] Answer: {answer} (Confidence: {confidence:.4f})")
        return answer, confidence

    def plot_an_image(image_path):
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
