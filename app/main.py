from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from app.model import get_vit_embedding
from app.autoencoder import ViTAutoencoder

app = FastAPI()
model = ViTAutoencoder()
model.load_state_dict(torch.load("models/autoencoder.pth", map_location=torch.device("cpu")))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # ✅ Save image to temp path for embedding function
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        emb = get_vit_embedding(temp_path)

        with torch.no_grad():
            recon = model(emb)
            loss = torch.nn.functional.mse_loss(recon, emb)
            score = loss.item()

        threshold = 0.05
        is_anomaly = score > threshold

        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly
        }

    except Exception as e:
        print("❌ EXCEPTION:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
