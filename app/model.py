import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image

# Load Hugging Face ViT and feature extractor globally
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()

# Function to get ViT embedding from an input PIL image
def get_vit_embedding(image: Image.Image) -> torch.Tensor:
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
    return embedding.squeeze()

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
