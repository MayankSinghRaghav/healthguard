import os
import requests

save_dir = "data/train/normal"
os.makedirs(save_dir, exist_ok=True)

image_urls = [
    # Original 3 working ones
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-b.jpg",
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg",

    # 7 more verified images (JPG or PNG)
    "https://raw.githubusercontent.com/agchung/ChestXray-CoronaDataset/master/Coronahack-Chest-XRay-Dataset/train/NORMAL/IM-0115-0001.jpeg",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-001-a1.png",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-002-a1.png",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-002-a2.png",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-003-b1.png",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-003-b2.png",
    "https://raw.githubusercontent.com/agchung/Figure1-COVID-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-004-b1.png"
]

for idx, url in enumerate(image_urls):
    filename = os.path.join(save_dir, f"xray_{idx+1}.jpg")
    print(f"Downloading {filename} ...")
    try:
        response = requests.get(url, timeout=10)
        with open(filename, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("✅ Download complete.")
