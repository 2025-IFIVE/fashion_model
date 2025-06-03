import os
import random
import numpy as np
import pandas as pd

from .color_utils import extract_dominant_hsv_cached, classify_color_relation
from .weather_rules import WEATHER_TYPE_COMBINATIONS, SLEEVE_RULES_BY_WEATHER, OUTER_CATEGORY_RULES_BY_WEATHER
from .model_loader import load_gmm_model

SPRING_IMAGE_ROOT = "C:/Users/11jud/Desktop/capstone/fitza_BE/fitza"

def get_cropped_image_path(item):
    cropped = item.get("croppedPath")
    if not cropped:
        raise ValueError("croppedPath가 존재하지 않습니다.")
    filename = os.path.basename(cropped)
    return os.path.join(SPRING_IMAGE_ROOT, "uploads", "cropped", filename)

def get_image_vec(item):
    full_path = get_cropped_image_path(item)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"이미지 없음: {full_path}")
    return extract_dominant_hsv_cached(full_path)

def is_valid_sleeve(item, weather):
    sleeve = item.get("sleeve")
    if sleeve is None:
        return False
    return sleeve in SLEEVE_RULES_BY_WEATHER.get(weather, [])

def is_valid_outer(item, weather):
    category = item.get("category")
    if category is None:
        return False
    return category in OUTER_CATEGORY_RULES_BY_WEATHER.get(weather, [])

def generate_recommendation(closet_items, weather: str, top_k: int = 10):
    categorized = {"상의": [], "하의": [], "아우터": [], "원피스": []}
    for item in closet_items:
        type_ = item.get("type")
        if type_ in categorized:
            categorized[type_].append(item)

    rules = WEATHER_TYPE_COMBINATIONS.get(weather)
    if not rules:
        raise ValueError(f"정의되지 않은 날씨 룰: {weather}")

    combos = []
    for rule in rules:
        if "원피스" in rule:
            for dress in categorized["원피스"]:
                if "아우터" in rule and categorized["아우터"]:
                    for outer in categorized["아우터"]:
                        if not is_valid_outer(outer, weather):
                            continue
                        combos.append([dress, outer])
                else:
                    combos.append([dress])
        else:
            for top in categorized["상의"]:
                if not is_valid_sleeve(top, weather):
                    continue
                for bottom in categorized["하의"]:
                    if "아우터" in rule and categorized["아우터"]:
                        for outer in categorized["아우터"]:
                            if not is_valid_outer(outer, weather):
                                continue
                            combos.append([top, bottom, outer])
                    else:
                        combos.append([top, bottom])

    if not combos:
        return None

    gmm, scaler = load_gmm_model()
    scored = []

    for combo in combos:
        vec = []
        slots = {"top": None, "bottom": None, "outer": None, "dress": None}
        for item in combo:
            if item["type"] == "상의":
                slots["top"] = item
            elif item["type"] == "하의":
                slots["bottom"] = item
            elif item["type"] == "아우터":
                slots["outer"] = item
            elif item["type"] == "원피스":
                slots["dress"] = item

        try:
            top_vec = get_image_vec(slots["top"]) if slots["top"] else np.array([0, 0, 0])
            bottom_vec = get_image_vec(slots["bottom"]) if slots["bottom"] else np.array([0, 0, 0])
            outer_vec = get_image_vec(slots["outer"]) if slots["outer"] else np.array([0, 0, 0])
            dress_vec = get_image_vec(slots["dress"]) if slots["dress"] else np.array([0, 0, 0])

            rel_tb = classify_color_relation(*top_vec, *bottom_vec) if slots["top"] and slots["bottom"] else -1
            rel_to = classify_color_relation(*top_vec, *outer_vec) if slots["top"] and slots["outer"] else -1
            rel_td = classify_color_relation(*top_vec, *dress_vec) if slots["top"] and slots["dress"] else -1

            vector = list(top_vec) + list(bottom_vec) + list(outer_vec) + list(dress_vec) + [rel_tb, rel_to, rel_td]
            df = pd.DataFrame([vector], columns=scaler.feature_names_in_)
            scaled = scaler.transform(df)

            probs = gmm.predict_proba(scaled)[0]
            cluster_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))

            scored.append((slots, confidence, cluster_idx))

        except Exception as e:
            print(f"조합 예측 실패: {e}")

    if not scored:
        return None

    scored.sort(key=lambda x: -x[1])
    slot_map, confidence, cluster_idx = random.choice(scored[:top_k])

    result_items = []
    for part in ["top", "bottom", "outer", "dress"]:
        if slot_map[part]:
            item = slot_map[part]
            result = {
                "category": item["type"],
                "id": item.get("id") or item.get("clothid"),
                "imageUrl": item["croppedPath"],
                "cluster": cluster_idx,
                "sleeve": item.get("sleeve") if item["type"] == "상의" else None,
                "outerCategory": item.get("category") if item["type"] == "아우터" else None
            }
            result_items.append(result)

    return {
        "confidence": round(confidence, 4),
        "cluster": cluster_idx,
        "items": result_items
    }
