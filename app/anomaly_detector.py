import torch
from app.autoencoder import ViTAutoencoder
from app.model import get_vit_embedding

model = ViTAutoencoder()
model.load_state_dict(torch.load("models/autoencoder.pth"))
model.eval()

def detect_anomaly(img_path, threshold=0.05):
    emb = get_vit_embedding(img_path)
    with torch.no_grad():
        recon = model(emb)
        error = torch.nn.functional.mse_loss(recon, emb)
    return error.item(), error.item() > threshold
