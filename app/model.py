# app/model.py

import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor

# Load Hugging Face ViT and feature extractor globally
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def load_model():
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.eval()
    return model

# Wrapper class to expose the final hidden states for Grad-CAM (simulate classification head)
class WrapViT(nn.Module):
    def __init__(self, vit_model):
        super(WrapViT, self).__init__()
        self.vit = vit_model
        # Add a dummy head for classification (needed for Grad-CAM)
        self.head = nn.Linear(self.vit.config.hidden_size, 2)

    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.head(cls_token)
        return logits

# Optional: feature extractor transform for consistency
def get_transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])
