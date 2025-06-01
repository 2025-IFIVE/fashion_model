import os
import io
import numpy as np
import cv2
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans

# 🔹 배경 제거 + 캐시
def remove_background_cached(image_path, save_dir="cache/cleaned"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    if os.path.exists(save_path):
        return Image.open(save_path).resize((100, 100))
    with open(image_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)
    img = Image.open(io.BytesIO(output_bytes)).convert("RGB").resize((100, 100))
    img.save(save_path)
    return img

# 🔹 dominant HSV + 캐시
def extract_dominant_hsv_cached(image_path, save_dir="cache/colors"):
    os.makedirs(save_dir, exist_ok=True)
    npy_path = os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + ".npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    # 🔄 배경 제거 생략! 바로 이미지 열기
    image = Image.open(image_path).convert("RGB").resize((100, 100))
    img_np = np.array(image)
    pixels = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)
    dominant_rgb = kmeans.cluster_centers_[0].astype(np.uint8)

    hsv = cv2.cvtColor(np.uint8([[dominant_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv.astype(float)
    vec = np.array([h / 360.0, s / 255.0, v / 255.0])
    np.save(npy_path, vec)
    return vec

# 🔹 색상 조화 판단
def classify_color_relation(h1, s1, v1, h2, s2, v2):
    hue_diff = abs(h1 - h2)
    hue_diff = min(hue_diff, 1 - hue_diff)
    sat_diff = abs(s1 - s2)
    val_diff = abs(v1 - v2)

    if hue_diff <= 0.05 and (sat_diff > 0.2 or val_diff > 0.2):
        return 0  # 톤온톤
    elif hue_diff > 0.15 and sat_diff <= 0.2 and val_diff <= 0.2:
        return 1  # 톤인톤
    elif hue_diff <= 0.1 and sat_diff <= 0.2 and val_diff <= 0.2:
        return 2  # 톤널
    elif 0.48 <= hue_diff <= 0.52:
        return 3  # 보색
    elif hue_diff <= 0.08:
        return 4  # 유사색
    else:
        return 5  # 기타
