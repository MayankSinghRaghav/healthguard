import os
import torch
from model import get_vit_embedding

def extract_embeddings(folder_path):
    embeddings = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            emb = get_vit_embedding(os.path.join(folder_path, file))
            embeddings.append(emb)
    return torch.stack(embeddings)

if __name__ == "__main__":
    embeddings = extract_embeddings("data/train/normal")
    torch.save(embeddings, "models/train_embeddings.pt")
