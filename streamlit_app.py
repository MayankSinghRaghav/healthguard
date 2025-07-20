import streamlit as st
import requests
import tempfile
from PIL import Image

st.set_page_config(page_title="HealthGuard: X-ray Anomaly Detector", layout="centered")

st.title("🩻 HealthGuard")
st.markdown("Upload a chest X-ray to detect anomalies.")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼️ Uploaded X-ray", use_column_width=True)

    # Save uploaded image to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Send image to FastAPI backend
    with open(tmp_path, "rb") as file:
        files = {"file": (uploaded_file.name, file, "image/jpeg")}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)

    # Display result
    if response.status_code == 200:
        result = response.json()
        st.subheader("🔎 Prediction")
        st.write(f"**Anomaly Score:** `{result['anomaly_score']:.4f}`")
        st.write(f"**Is Anomaly:** `{result['is_anomaly']}`")
    else:
        st.error(f"❌ Server returned error {response.status_code}")
else:
    st.info("👆 Upload an image to get started.")
