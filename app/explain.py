# app/explain.py
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.model import load_model, WrapViT

def generate_gradcam(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and wrap model
    model = load_model()
    model.eval().to(device)
    wrapped_model = WrapViT(model)
    wrapped_model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Prepare CAM
    target_layers = [wrapped_model.vit.encoder.layer[-1].output]
    cam = AblationCAM(model=wrapped_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # Forward once to get predicted class
    with torch.no_grad():
        outputs = wrapped_model(input_tensor)
        predicted_class = outputs.argmax().item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]

    # Overlay CAM on image
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return Image.fromarray(cam_image)
