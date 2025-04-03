import os
import torch
import models.model
from utils.config import ConfigLoader
from timm.models import create_model
from utils.dataset import Process, ViVQADataset, ViVQAProcessor
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
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
            state_dict = torch.load(
                model_path, map_location=device, weights_only=True)

            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully.")
        else:
            print("Checkpoint path does not exist")

        return model.to(device).eval()

    def get_test_dataset(self, test_path):
        processor = Process()
        image_path = self._paths_cfg["image_path"]
        ans_path = self._paths_cfg["ans_path"]
        df_test = pd.read_csv(test_path, index_col=0)
        test_dataset = ViVQADataset(df_test, processor, image_path, ans_path)
        return test_dataset

    def predict_test_dataset(self, model, test_path, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        test_dataset = self.get_test_dataset(test_path)
        dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        confidences = []
        ground_truths = []
        for batch in tqdm(dataloader, desc="Running predictions", unit="sample"):
            images = batch["image"].to(device, dtype=torch.float32)
            questions = batch["question"].to(device)
            masks = batch["padding_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            with torch.no_grad():
                logits = model(image=images, question=questions,
                               padding_mask=masks).logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1)

            ground_truth = [batch["answers"][i] for i in range(len(labels))]

            for i in range(len(pred_idx)):
                pred = pred_idx[i].item()
                confidence = probs[i][pred].item()
                answer = batch["answers"][i]
                predictions.append(answer)
                confidences.append(confidence)
                ground_truths.append(ground_truth[i])

        self.save_predictions(predictions, confidences,
                              ground_truths, test_dataset)

    def save_predictions(self, predictions, confidences, ground_truths, test_dataset):
        questions, img_ids = zip(*self._extract_metadata(test_dataset))

        df = pd.DataFrame({
            "question": questions,
            "img_id": img_ids,
            "answer": predictions,
            "confidence": confidences,
            "ground_truth": ground_truths
        })

        os.makedirs(self._paths_cfg["prediction_path"], exist_ok=True)
        subset_id = self._cfg.get("training")["subset_id"]
        filename = f"answers-{subset_id}.csv" if subset_id is not None else "answers.csv"
        output_path = os.path.join(
            self._paths_cfg["prediction_path"], filename)

        df.to_csv(output_path, encoding="utf-8")
        print(f"[INFO] Predictions saved to {output_path}")

    def _extract_metadata(self, test_dataset):
        for idx in range(len(test_dataset)):
            metadata = test_dataset.get_sample_metadata(idx)
            if metadata:
                yield metadata["question"], metadata["img_id"]

    def predict_sample(self, model, image_path, question):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"[INFO] Question: {question}")

        vivqa_processor = ViVQAProcessor(self._paths_cfg)
        sample = vivqa_processor.process_sample(image_path, question)
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
        self.plot_an_image(image_path)

    def plot_an_image(self, image_path):
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
