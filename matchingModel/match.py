import os
import requests
from PIL import Image
from io import BytesIO
from ultralyticsplus import YOLO
import torch
import clip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("kesimeg/yolov8n-clothing-detection")

SPRING_WARDROBE_API = "http://localhost:8080/api/internal/wardrobe"

def download_image_from_url(url):
    try:
        response = requests.get(url)
        print(f"🔗 요청 URL: {url} | 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"⚠️ 다운로드 실패 ❌ -> 응답 내용: {response.text[:200]}")
            return None

        return Image.open(BytesIO(response.content))

    except Exception as e:
        print(f"❌ 예외 발생: {e} | 요청 URL: {url}")
        return None


def match_image_against_db(user_id, image_path, top_n=3):
    print("📡 옷장 조회 요청:", f"{SPRING_WARDROBE_API}/{user_id}")
    try:
        wardrobe_response = requests.get(f"{SPRING_WARDROBE_API}/{user_id}")
        wardrobe_response.raise_for_status()
        wardrobe_items = wardrobe_response.json()
    except Exception as e:
        return {"error": f"Spring API 요청 실패: {e}"}

    if not wardrobe_items:
        return {"error": "등록된 옷 이미지가 없습니다."}

    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    results = []
    for item in wardrobe_items:
        image_url = "http://localhost:8080" + (item.get("croppedPath") or item.get("imagePath"))
        cloth_id = item.get("clothId")
        image_obj = download_image_from_url(image_url)
        if image_obj is None:
            continue

        try:
            closet_img = preprocess(image_obj).unsqueeze(0)
            with torch.no_grad():
                closet_feat = clip_model.encode_image(closet_img)
            sim = cosine_similarity(image_features.cpu(), closet_feat.cpu())[0][0]
            results.append({
                "clothId": cloth_id,
                "imageUrl": image_url,
                "similarity": float(sim)
            })
        except Exception as e:
            print(f"⚠️ 이미지 처리 오류: {image_url}, {e}")

    top_matches = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]
    return {
    "matchedImages": [item["imageUrl"] for item in top_matches],
    "labels": [item["clothId"] for item in top_matches],
    "scores": [item["similarity"] for item in top_matches]
}

