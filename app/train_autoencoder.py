import torch
from torch.utils.data import DataLoader, TensorDataset
from app.autoencoder import ViTAutoencoder

# Load pre-extracted ViT embeddings
train_data = torch.load("models/train_embeddings.pt")

# Wrap into DataLoader
dataset = TensorDataset(train_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Init model
model = ViTAutoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(20):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]  # Each x is a 768-dim ViT vector
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/20 | Loss: {total_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "models/autoencoder.pth")
print("✅ Autoencoder training complete.")
