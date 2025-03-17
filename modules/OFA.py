import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers.utils.generic import ModelOutput


@dataclass
class SelectiveViVQAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    confidence: torch.FloatTensor = None
    confidence_loss: Optional[torch.FloatTensor] = None


class SelectivePredictor(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_1: int = 768,
        hidden_2: int = 768,
        dropout: float = 0.1,
        features: str = "pooled_text+pooled_img+prob",
    ):
        super().__init__()
        self.features = features.split("+")
        
        # Calculate input dimension based on selected features
        input_dim = 0
        if "pooled_text" in self.features:
            input_dim += hidden_size
        if "pooled_img" in self.features:
            input_dim += hidden_size
        if "cls_rep" in self.features:
            input_dim += hidden_size
        if "prob" in self.features:
            input_dim += 1
        if "logits" in self.features:
            input_dim += hidden_size
        
        # MLP for confidence prediction
        self.selective_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )
        
        self.init_weights()
        
    def init_weights(self):
        for module in self.selective_predictor:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = []
        for feature_name in self.features:
            if feature_name in features_dict:
                feature = features_dict[feature_name]
                if feature_name == "prob" and feature.dim() == 1:
                    feature = feature.unsqueeze(1)
                inputs.append(feature)
            else:
                raise ValueError(f"Feature {feature_name} not found in input features")
        
        x = torch.cat(inputs, dim=1)
        
        confidence = self.selective_predictor(x)
        
        return confidence


class SelectiveBEiT3ForVietnameseVQA(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size: int = 768,
        selective_hidden_1: int = 768,
        selective_hidden_2: int = 768,
        selective_dropout: float = 0.1,
        selective_features: str = "pooled_text+pooled_img+prob+cls_rep",
        freeze_base_model: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add selective predictor
        self.selective_predictor = SelectivePredictor(
            hidden_size=hidden_size,
            hidden_1=selective_hidden_1,
            hidden_2=selective_hidden_2,
            dropout=selective_dropout,
            features=selective_features,
        )
        
    def forward(
        self,
        image, 
        question, 
        padding_mask, 
        labels=None,
        confidence_labels=None,
        confidence_loss_weight: float = 1.0,
        **kwargs
    ):
        # Get base model outputs
        outputs = self.base_model(
            image=image,
            question=question,
            padding_mask=padding_mask,
            labels=labels,
            **kwargs
        )
        
        # Extract features for selective predictor
        features_dict = {}
        
        # Get the encoder output and pooled representations from the base model
        encoder_outputs = self.base_model.beit3(
            textual_tokens=question.squeeze(dim=1),
            visual_tokens=image,
            text_padding_position=padding_mask.squeeze(dim=1),
        )
        
        x = encoder_outputs["encoder_out"]
        multiway_split_position = encoder_outputs["multiway_split_position"]
        
        # Get text and image embeddings
        img_embed = x[:, :multiway_split_position, :]
        text_embed = x[:, multiway_split_position:, :]
        
        # Extract pooled representations
        text_embed_pooled = text_embed.max(dim=1).values
        img_embed_pooled = img_embed.max(dim=1).values
        
        # Get cls representation
        cls_rep = self.base_model.pooler(x)
        
        # Add features to dictionary
        features_dict["pooled_text"] = text_embed_pooled
        features_dict["pooled_img"] = img_embed_pooled
        features_dict["cls_rep"] = cls_rep
        
        # Add probability of prediction if we have logits
        if hasattr(outputs, "logits"):
            features_dict["logits"] = outputs.logits
            probs = F.softmax(outputs.logits, dim=-1)
            features_dict["prob"] = torch.max(probs, dim=1).values
        
        # Forward pass through selective predictor
        confidence = self.selective_predictor(features_dict)
        
        # Calculate confidence loss if labels are provided
        confidence_loss = None
        if confidence_labels is not None:
            confidence_loss = F.binary_cross_entropy_with_logits(
                confidence.squeeze(), confidence_labels.float()
            )
        
        # Calculate total loss
        loss = outputs.loss
        if confidence_loss is not None and loss is not None:
            loss = loss + confidence_loss_weight * confidence_loss
        
        return SelectiveViVQAOutput(
            loss=loss,
            logits=outputs.logits,
            confidence=confidence.squeeze(),
            confidence_loss=confidence_loss,
        )
    
    def predict_with_confidence(
        self,
        image, 
        question, 
        padding_mask, 
        confidence_threshold: float = 0.5,
        **kwargs
    ):

        # Get model outputs
        outputs = self.forward(image, question, padding_mask, **kwargs)
        
        # Get answer prediction and confidence
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1)
        confidence = torch.sigmoid(outputs.confidence)
        
        # Decide whether to abstain
        abstained = confidence < confidence_threshold
        
        return {
            "prediction": pred_idx,
            "confidence": confidence,
            "abstained": abstained,
            "probs": probs
        }


def convert_base_model_to_selective(
    base_model,
    hidden_size: int = 768,
    selective_hidden_1: int = 768,
    selective_hidden_2: int = 768,
    selective_dropout: float = 0.1,
    selective_features: str = "pooled_text+pooled_img+prob+cls_rep",
    freeze_base_model: bool = True,
):

    selective_model = SelectiveBEiT3ForVietnameseVQA(
        base_model=base_model,
        hidden_size=hidden_size,
        selective_hidden_1=selective_hidden_1,
        selective_hidden_2=selective_hidden_2,
        selective_dropout=selective_dropout,
        selective_features=selective_features,
        freeze_base_model=freeze_base_model,
    )
    
    return selective_model 