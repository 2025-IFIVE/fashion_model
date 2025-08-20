# -*- coding: utf-8 -*-
"""
전신 → (YOLO 크롭) → (CLIP 임베딩 매칭 Top-N) → (필요시 재랭크 확장 가능)
→ Top-K 중 Top1씩 반환 (main.py에서 중복 제거 로직 수행)
"""

import os
import io
import json
import math
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional


import numpy as np
import cv2
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip

# YOLO (Ultralytics)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from ultralytics import YOLO
    except Exception:
        YOLO = None

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────
# 환경설정 / 상수
SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://localhost:8080")
# 기존 코드와 동일한 엔드포인트 패턴 유지: /api/internal/wardrobe/{userId}
SPRING_WARDROBE_API = os.getenv(
    "SPRING_WARDROBE_API",
    f"{SPRING_BASE_URL}/api/internal/wardrobe"
)

CFG = {
    # YOLO
    "YOLO_WEIGHTS": os.getenv("YOLO_WEIGHTS", "weights/best.pt"),
    "YOLO_CONF": float(os.getenv("YOLO_CONF", "0.25")),  # 약간 올림(노이즈 감소)
    "YOLO_IMGSZ": int(os.getenv("YOLO_IMGSZ", "768")),
    "YOLO_IOU_TH": float(os.getenv("YOLO_IOU_TH", "0.60")),
    "YOLO_DRESS_TH": float(os.getenv("YOLO_DRESS_TH", "0.35")),  # 외부 모델 사용 시 과도한 필터 방지
    "YOLO_NO_SUPPRESS": os.getenv("YOLO_NO_SUPPRESS", "false").lower() == "true",

    # CLIP
    "CLIP_MODEL_NAME": os.getenv("CLIP_MODEL_NAME", "ViT-B-16"),
    "CLIP_PRETRAINED": os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b88k"),
    "EMBED_CACHE_DIR": os.getenv("EMBED_CACHE_DIR", "clip_cache"),

    # Top-N / Top-K (여기선 각 파츠별 Top1만 사용, 필요시 확장)
    "TOPN": int(os.getenv("TOPN", "150")),
    "TOPK": int(os.getenv("TOPK", "1")),
}

# YOLO 클래스 정규화
def _norm_label(lbl: str) -> str:
    s = str(lbl).lower()
    if "dress" in s or "원피스" in s:
        return "dress"
    if "outer" in s or "아우터" in s or "jacket" in s or "coat" in s:
        return "outer"
    if "bottom" in s or "하의" in s or "pants" in s or "skirt" in s:
        return "bottom"
    if "top" in s or "상의" in s or "upper" in s:
        return "top"
    # 그 외 (shoes, bags 등)도 그대로 반환
    return s

# ─────────────────────────────────────────────────────────
# 경로/입출력 유틸
def to_absolute_url(path: str, base: str = SPRING_BASE_URL) -> str:
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return base.rstrip("/") + path

def imread_any(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
    return img

def _download_pil(url: str) -> Image.Image:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ─────────────────────────────────────────────────────────
# 전역 모델 로더
_yolo_model = None
_clip_model = None
_clip_preprocess = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        if YOLO is None:
            raise RuntimeError("ultralytics가 설치되지 않았습니다. pip install ultralytics")
        _yolo_model = YOLO(CFG["YOLO_WEIGHTS"])

def _load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CFG["CLIP_MODEL_NAME"], pretrained=CFG["CLIP_PRETRAINED"], device=DEVICE
        )
        _clip_model = _clip_model.eval()

# ─────────────────────────────────────────────────────────
# CLIP 임베딩
def _embed_pil(img: Image.Image) -> np.ndarray:
    _load_clip()
    x = _clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = _clip_model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy()

def _embed_bgr(img_bgr: np.ndarray) -> np.ndarray:
    _load_clip()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return _embed_pil(pil)

def _cosine_sim_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # vecs shape: (1, d)
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.dot(v1, v2.T) / denom)

# ─────────────────────────────────────────────────────────
# 드레스 vs 상·하의 세트 선택 
def _best_conf_from_dets(dets: List[Dict], label: str) -> Optional[float]:
    vals = [float(d.get("conf", 0.0)) for d in dets if d.get("label") == label]
    return max(vals) if vals else None

def _enforce_set_choice_from_dets(dets: List[Dict]) -> List[Dict]:
    
    has_d = any(d.get("label") == "dress" for d in dets)
    has_t = any(d.get("label") == "top"   for d in dets)
    has_b = any(d.get("label") == "bottom" for d in dets)

    if not (has_d and has_t and has_b):
        return dets

    bd = _best_conf_from_dets(dets, "dress")
    bt = _best_conf_from_dets(dets, "top")
    bb = _best_conf_from_dets(dets, "bottom")
    if bd is None or bt is None or bb is None:
        return dets

    score_dress = bd
    score_tb = math.sqrt(bt * bb)

    keep = {"outer"}  # 아우터는 항상 유지
    if score_dress >= score_tb:
        keep.add("dress")
    else:
        keep.update({"top", "bottom"})

    return [d for d in dets if d.get("label") in keep]
# ─────────────────────────────────────────────────────────
# YOLO 크롭
def _yolo_detect_and_crops(image_path: str) -> Tuple[List[Dict], np.ndarray]:
    
    _load_yolo()
    res = _yolo_model.predict(
        source=image_path,
        conf=CFG["YOLO_CONF"],
        iou=CFG["YOLO_IOU_TH"],
        imgsz=CFG["YOLO_IMGSZ"],
        save=False,
        save_txt=False,
        verbose=False
    )
    img_bgr = imread_any(image_path)
    names = _yolo_model.names if hasattr(_yolo_model, "names") else {}
    detections = []

    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten().tolist()
            confv = float(b.conf.cpu().numpy().item())
            clsid = int(b.cls.cpu().numpy().item())
            lbl = _norm_label(names.get(clsid, str(clsid)))
            if lbl == "dress" and confv < CFG["YOLO_DRESS_TH"]:
                continue
            detections.append({"label": lbl, "conf": confv, "xyxy": xyxy})

    # 라벨별 최고 신뢰도만 남기고 싶으면 아래 블록 활성화
    if not CFG["YOLO_NO_SUPPRESS"]:
        best = {}
        for d in detections:
            k = d["label"]
            if (k not in best) or (d["conf"] > best[k]["conf"]):
                best[k] = d
        detections = list(best.values())

    return detections, img_bgr

def _crop_from_xyxy(img_bgr: np.ndarray, xyxy: List[float]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]

# ─────────────────────────────────────────────────────────
# 옷장 불러오기 & 임베딩
def _fetch_wardrobe_items(user_id: int) -> List[Dict]:
    # 기존 코드와 동일한 패턴: /api/internal/wardrobe/{userId}
    url = f"{SPRING_WARDROBE_API}/{user_id}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"[Spring 오류] 옷장 조회 실패: {r.status_code} - {r.text}")
    return r.json()

def _build_wardrobe_embeddings(wardrobe_items: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    
    embs = []
    info = []
    for item in wardrobe_items:
        raw_path = item.get("imagePath", "")
        if not raw_path:
            continue
        url = to_absolute_url(raw_path, SPRING_BASE_URL)
        try:
            pil = _download_pil(url)
        except Exception as e:
            print(f"[wardrobe] 다운로드 실패: {url} | {e}")
            continue
        try:
            feat = _embed_pil(pil)  # (1, d)
            embs.append(feat[0])
            info.append({
                "clothId": item.get("clothId"),
                "imagePath": url,
                "type": item.get("type", "UNKNOWN")
            })
        except Exception as e:
            print(f"[wardrobe] 임베딩 실패: {url} | {e}")
            continue

    if not embs:
        raise RuntimeError("옷장 이미지 임베딩 생성 실패 (다운로드/임베딩 실패)")

    return np.stack(embs, axis=0), info

# ─────────────────────────────────────────────────────────
# 공개 함수: main.py에서 import
def match_image_against_db(user_id: int, image_path: str, top_n: int = 1) -> Dict:
    
    # 0) 유저 옷장 가져오기 & 임베딩 구성
    wardrobe_items = _fetch_wardrobe_items(user_id)
    wardrobe_embs, wardrobe_info = _build_wardrobe_embeddings(wardrobe_items)  # [N, d], list len N
    # 1) YOLO 감지
    detections, img_bgr = _yolo_detect_and_crops(image_path)

    detections = _enforce_set_choice_from_dets(detections)

    matches = []
    for i, det in enumerate(detections):
        xyxy = det["xyxy"]
        label = det["label"]

        crop = _crop_from_xyxy(img_bgr, xyxy)
        if crop is None:
            continue

        # 2) 파츠 크롭 임베딩
        try:
            part_emb = _embed_bgr(crop)  # (1, d)
            q = part_emb[0]
        except Exception as e:
            print(f"[part] 임베딩 실패 (index={i}, label={label}) | {e}")
            continue

        # 3) 코사인 유사도 → Top-N 중 Top1 선택 (main에서 중복 제거 수행)
        # (wardrobe_embs shape: [N, d])
        # 빠르게 벡터 유사도 계산
        norms = (np.linalg.norm(wardrobe_embs, axis=1) * (np.linalg.norm(q) + 1e-8) + 1e-8)
        sims = (wardrobe_embs @ q) / norms  # [N,]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        win = wardrobe_info[best_idx]
        matches.append({
            "partIndex": i,
            "partLabel": label,
            "matchedClothId": win["clothId"],
            "matchedImagePath": win["imagePath"],
            "matchedType": win.get("type", "UNKNOWN"),
            "similarity": best_sim
        })

    result = {
        "userId": user_id,
        "numDetectedParts": len(detections),
        "matches": matches
    }
    return result
