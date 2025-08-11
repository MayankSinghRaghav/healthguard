# 🩺 HealthGuard: Autonomous Anomaly Detection in Medical Imaging

HealthGuard is a lightweight AI-based tool that helps detect anomalies in medical X-ray images. It uses a pre-trained ViT (Vision Transformer) model to extract deep visual embeddings and provides a simple interface for triaging urgent cases.

---

## 📌 Features

- 🧠 Uses Vision Transformer (ViT) for medical image embedding.
- 🖼️ Upload and analyze chest X-ray images.
- 🧪 Returns anomaly detection embeddings (can be extended with classifiers or anomaly scores).
- 🌐 Interactive UI built with Streamlit.
- 💻 Works offline with pre-loaded model weights.

---
Backend run server : uvicorn app.main:app --reload
Frontend Dashboard : streamlit run streamlit_app.py

## 🚀 Installation

```bash
git clone https://github.com/your-username/healthguard.git
cd healthguard
python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on macOS/Linux

pip install -r requirements.txt
