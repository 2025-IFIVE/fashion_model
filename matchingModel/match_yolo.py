import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from ultralyticsplus import YOLO
import torch
import cv2
import random

from inference import preprocess_image, run_inference
from model_loader import load_models

def visualize_yolo_results(image_path, results, save_path="output_with_boxes.jpg"):
    image = cv2.imread(image_path)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = yolo_model.names[cls_id]
        if class_name not in ["clothing", "shoes", "accessories"]:  # 원하는 객체만 표시
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        color = [int(c) for c in np.random.randint(0, 255, size=3)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(save_path, image)
    print(f"🖼️ YOLO 탐지 시각화 저장됨: {save_path}")

# 모델 로드
print("모델 로딩 중...")
type_model, attr_model, style_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("kesimeg/yolov8n-clothing-detection")
print("모델 로딩 완료")

SPRING_WARDROBE_API = "http://localhost:8080/api/internal/wardrobe"

# DB 속성 dict 구성
def extract_db_attributes(item):
    keys = ["category", "length", "sleeve", "neckline", "neck", "fit", "color", "material", "detail", "print", "style", "substyle"]
    attr_dict = {}
    for key in keys:
        val = item.get(key)
        if val:
            attr_dict[key] = val.split(",") if key in ["color", "material", "detail", "print"] else val
    return attr_dict

# 유사도 계산
def compute_attribute_similarity(predicted, db_item):
    KEY_MAPPING = {
        "카테고리": "category",
        "기장": "length",
        "소매기장": "sleeve",
        "넥라인": "neckline",
        "칼라": "neck",
        "핏": "fit",
        "색상": "color",
        "소재": "material",
        "디테일": "detail",
        "프린트": "print",
        "스타일": "style",
        "서브스타일": "substyle"
    }

    score = 0
    total = 0
    match_log = []

    for pred_key, pred_val in predicted.items():
        db_key = KEY_MAPPING.get(pred_key)
        if not db_key:
            continue
        db_val = db_item.get(db_key)
        if not db_val:
            continue

        if isinstance(pred_val, list):
            db_val_list = [db_val.strip()] if isinstance(db_val, str) else db_val
            common = set(pred_val) & set(db_val_list)
            union = set(pred_val) | set(db_val_list)
            sim = len(common) / len(union) if union else 0
            score += sim
            match_log.append(f"{pred_key}: {sim:.2f}")
        else:
            match = str(pred_val).strip() == str(db_val).strip()
            score += 1.0 if match else 0
            match_log.append(f"{pred_key}: {'1' if match else '0'}")

        total += 1

    final_score = score / total if total else 0.0
    print(f"속성 일치 요약: {' | '.join(match_log)}")
    print(f"유사도 점수: {final_score:.3f}")
    return final_score





# 전체 매칭 로직
def match_image_against_db_v2(user_id, image_path, top_n=3):
    print(f"\n/match_image_against_db_v2 시작 (user_id={user_id})")

    try:
        wardrobe_response = requests.get(f"{SPRING_WARDROBE_API}/{user_id}")
        wardrobe_response.raise_for_status()
        wardrobe_items = wardrobe_response.json()
    except Exception as e:
        print(f"❌ Spring 통신 실패: {e}")
        return {"error": f"Spring API 요청 실패: {e}"}

    if not wardrobe_items:
        print("⚠️ 옷장이 비어 있습니다.")
        return {"error": "등록된 옷 이미지가 없습니다."}

    original_img = cv2.imread(image_path)
    results = yolo_model(image_path)[0]
    visualize_yolo_results(image_path, results)
    print(f"📸 이미지에서 객체 {len(results.boxes)}개 탐지됨")

    all_matches = []

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        raw_label = yolo_model.names[cls_id]
        if raw_label != "clothing":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = original_img[y1:y2, x1:x2]

        tensor = preprocess_image(BytesIO(cv2.imencode(".png", cropped_img)[1].tobytes()))
        result = run_inference(tensor, type_model, attr_model, style_model)
        part = result["의류종류"]
        predicted_attrs = result["속성"].get(part, {})

        # 🔍 분석 결과 요약 출력
        print(f"\n👕 [탐지 #{i+1}] 분석된 의류 종류: {part}")
        print("분석된 속성:")
        for k, v in predicted_attrs.items():
            print(f"   · {k}: {v}")

        scores = []
        for item in wardrobe_items:
            if item.get("type") != part:
                continue
            db_attrs = extract_db_attributes(item)
            sim = compute_attribute_similarity(predicted_attrs, db_attrs)
            match_keys = []
            for key, value in predicted_attrs.items():
                if key in db_attrs:
                    if isinstance(value, list) and isinstance(db_attrs[key], list):
                        if set(value) & set(db_attrs[key]):
                            match_keys.append(key)
                    elif value == db_attrs[key]:
                        match_keys.append(key)

            scores.append({
                "similarity": float(sim),
                "matchKeys": match_keys,
                "imageUrl": "http://localhost:8080" + item.get("croppedPath", item.get("imagePath", "")),
                "clothId": item.get("clothid")
            })

        print("\n🆚 옷장 비교 결과:")
        for item in sorted(scores, key=lambda x: x["similarity"], reverse=True)[:top_n]:
            print(f" 👗 옷 ID: {item['clothId']}")
            print(f"    └ 유사도: {item['similarity']:.2f}")
            print(f"    └ 일치 속성: {', '.join(item['matchKeys']) if item['matchKeys'] else '없음'}")

        if scores:
            top_matches = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:top_n]
            all_matches.extend(top_matches)

    all_matches = sorted(all_matches, key=lambda x: x["similarity"], reverse=True)[:top_n]

    print("\n최종 Top 추천 결과:")
    for idx, m in enumerate(all_matches, 1):
        print(f" {idx}. 👕 옷 ID={m['clothId']} | 유사도={m['similarity']:.2f} | 일치: {', '.join(m['matchKeys'])}")

    return {
        "matchedImages": [m["imageUrl"] for m in all_matches],
        "labels": [m["clothId"] for m in all_matches],
        "scores": [m["similarity"] for m in all_matches]
    }

