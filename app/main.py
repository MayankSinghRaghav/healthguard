from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from app.model import get_vit_embedding
from app.autoencoder import ViTAutoencoder

app = FastAPI()

# Load trained autoencoder model
model = ViTAutoencoder()
model.load_state_dict(torch.load("models/autoencoder.pth", map_location=torch.device("cpu")))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Extract ViT embedding
        emb = get_vit_embedding(image).unsqueeze(0)  # Shape: (1, 768)

        # Get reconstruction and compute anomaly score
        with torch.no_grad():
            recon = model(emb)
            loss = torch.nn.functional.mse_loss(recon, emb)
            score = loss.item()

        # Threshold for anomaly
        threshold = 0.05
        is_anomaly = score > threshold

        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly
        }

    except Exception as e:
        print("❌ EXCEPTION:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
