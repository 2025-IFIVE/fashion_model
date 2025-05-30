import os
import torch
import clip
import numpy as np
from PIL import Image
from ultralyticsplus import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# ✅ PyTorch 2.6+ 대응: 안전하게 로드할 글로벌 클래스 등록
from torch.serialization import safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential  # ← 에러 메시지에 따라 추가
# 필요한 경우 여기서 다른 nn 모듈도 추가 가능

# YOLO 모델 경로 (Roboflow 모델 or 로컬 .pt)
YOLO_MODEL_PATH = "kesimeg/yolov8n-clothing-detection"

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ✅ YOLO 모델 로딩 시 safe_globals context로 감쌈
with safe_globals([DetectionModel, Sequential]):
    yolo_model = YOLO(YOLO_MODEL_PATH)

def match_image_against_db(user_id, image_path):
    """
    전신 이미지와 옷장 이미지 간 유사도 분석 (CLIP 기반).
    가장 유사한 옷 이미지 반환.
    """
    # 입력 이미지 전처리 및 임베딩
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    # 유저 옷장 이미지 임베딩
    closet_dir = f"./user_closets/{user_id}"
    similarities = []

    for fname in os.listdir(closet_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        closet_path = os.path.join(closet_dir, fname)
        closet_img = preprocess(Image.open(closet_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            closet_feat = clip_model.encode_image(closet_img)

        sim = cosine_similarity(image_features.cpu(), closet_feat.cpu())[0][0]
        similarities.append((fname, sim))

    # 유사도 기반 가장 유사한 이미지 선택
    if similarities:
        best_match = max(similarities, key=lambda x: x[1])
        return {
            "best_match": best_match[0],
            "similarity": float(best_match[1])
        }
    else:
        return {"error": "No images found in closet."}
