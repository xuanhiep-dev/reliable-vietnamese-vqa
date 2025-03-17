#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Add the project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

@dataclass
class ViVQAOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class DummyPreprocessor:
    """A dummy preprocessor that mimics the behavior of the real one for testing."""
    def __init__(self):
        pass
    
    def process_image(self, image):
        """Process an image into a tensor."""
        # Return a dummy tensor with the expected shape
        return torch.randn(1, 3, 224, 224)
    
    def process_text(self, text):
        """Process text into a token sequence."""
        # Return dummy tensors with expected shapes
        return torch.randint(0, 1000, (1, 32)), torch.zeros(1, 32).bool()

class DummyTextEmbedding(nn.Module):
    """A dummy text embedding module."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 768)
        
    def forward(self, tokens, attention_mask):
        return self.embedding(tokens)

class DummyVisionEmbedding(nn.Module):
    """A dummy vision embedding module."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        
    def forward(self, image):
        batch_size = image.size(0)
        x = self.conv(image)
        return x.view(batch_size, 768, -1).permute(0, 2, 1)

class DummyEncoder(nn.Module):
    """A simplified encoder that doesn't rely on torchscale."""
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=768, num_heads=2, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(768, 768 * 4),
            nn.ReLU(),
            nn.Linear(768 * 4, 768)
        )
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        
    def forward(self, x, padding_mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=padding_mask
        )
        x = x + attn_out
        x = self.norm1(x)
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return {"encoder_out": x}

class DummyViVQAModel(nn.Module):
    """A simplified ViVQA model for testing."""
    def __init__(self, num_classes=50):
        super().__init__()
        self.text_embed = DummyTextEmbedding()
        self.vision_embed = DummyVisionEmbedding()
        self.encoder = DummyEncoder()
        
        self.pooler = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.Tanh()
        )
        
        self.head = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.LayerNorm(768 * 2),
            nn.GELU(),
            nn.Linear(768 * 2, num_classes),
        )
    
    def forward(self, image, question, padding_mask, labels=None):
        # Process text and image embeddings
        text_embeds = self.text_embed(question, padding_mask)
        image_embeds = self.vision_embed(image)
        
        # Combine embeddings
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        combined_padding = torch.cat([
            torch.zeros(image_embeds.shape[0], image_embeds.shape[1], dtype=torch.bool, device=padding_mask.device),
            padding_mask
        ], dim=1)
        
        # Encode
        outputs = self.encoder(combined, combined_padding)
        x = outputs["encoder_out"]
        
        # Pool and predict
        cls_rep = self.pooler(x[:, 0, :])
        logits = self.head(cls_rep)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return ViVQAOutput(
            loss=loss,
            logits=logits
        )

def dummy_vivqa_model(pretrained=False, num_classes=50, **kwargs):
    """Create a dummy ViVQA model for testing."""
    model = DummyViVQAModel(num_classes=num_classes)
    return model 