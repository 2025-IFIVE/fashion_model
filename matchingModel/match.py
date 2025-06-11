def to_absolute_url(path):
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return "http://localhost:8080" + path

import os
import requests
import torch
import clip
import numpy as np
from PIL import Image
from ultralyticsplus import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# 모델 로드 (한 번만)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("kesimeg/yolov8n-clothing-detection")

SPRING_WARDROBE_API = "http://localhost:8080/api/internal/wardrobe"
SPRING_BASE_URL = "http://localhost:8080"

def to_absolute_url(path):
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return SPRING_BASE_URL + path

def download_image_from_url(url):
    try:
        response = requests.get(url)
        print(f"🔗 요청 URL: {url} | 상태 코드: {response.status_code}")
        if response.status_code != 200:
            print(f"⚠️ 다운로드 실패 ❌ -> 응답 내용: {response.text[:200]}")
            return None
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ 예외 발생: {e} | 요청 URL: {url}")
        return None

def match_image_against_db(user_id: int, image_path: str, top_n=1):
    print(f"\n[🚀] 매칭 시작 - userId: {user_id}")

    spring_url = f"{SPRING_WARDROBE_API}/{user_id}"
    response = requests.get(spring_url)
    if response.status_code != 200:
        raise Exception(f"[Spring 오류] 옷장 조회 실패: {response.status_code} - {response.text}")

    wardrobe_items = response.json()
    print(f"👚 유저 옷 개수: {len(wardrobe_items)}")

    if not wardrobe_items:
        raise Exception("❌ 유저의 옷장이 비어 있습니다.")

    wardrobe_embeddings, wardrobe_info = [], []
    for item in wardrobe_items:
        full_url = to_absolute_url(item["imagePath"])
        print(f"📦 옷 이미지 URL: {full_url}")
        img = download_image_from_url(full_url)
        if img is None:
            print(f"⚠️ 이미지 다운로드 실패 → 건너뜀: {full_url}")
            continue
        try:
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
            wardrobe_embeddings.append(emb.cpu().numpy())
            wardrobe_info.append({
                "clothId": item["clothId"],
                "imagePath": full_url,
                "type": item.get("type", "UNKNOWN")
            })
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패 → 건너뜀: {full_url} | 오류: {e}")

    if not wardrobe_embeddings:
        raise Exception("❌ 옷장 이미지 임베딩 생성 실패")

    wardrobe_embeddings = np.vstack(wardrobe_embeddings)

    original = Image.open(image_path).convert("RGB")
    results = yolo_model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    labels = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    part_matches = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped = original.crop((x1, y1, x2, y2))

        tensor = preprocess(cropped).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        part_feat = emb.cpu().numpy()

        sims = cosine_similarity(part_feat, wardrobe_embeddings)[0]
        best_idx = np.argmax(sims)

        part_matches.append({
            "partIndex": i,
            "partLabel": class_names[labels[i]],
            "matchedClothId": wardrobe_info[best_idx]["clothId"],
            "matchedImagePath": wardrobe_info[best_idx]["imagePath"],
            "matchedType": wardrobe_info[best_idx]["type"],
            "similarity": float(sims[best_idx])
        })

        print(f"👕 파츠 {i+1} ({class_names[labels[i]]}) → clothId {wardrobe_info[best_idx]['clothId']} (유사도 {sims[best_idx]:.2f})")

    return {
        "userId": user_id,
        "numDetectedParts": len(boxes),
        "matches": part_matches
    }
