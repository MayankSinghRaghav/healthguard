import requests

url = "http://127.0.0.1:8000/predict/"
file_path = "data/train/normal/xray_1.jpg"  # Change if needed

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("✅ Response:", response.status_code)
print("📊 Raw Text:", response.json())
