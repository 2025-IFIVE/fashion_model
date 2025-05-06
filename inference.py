from PIL import Image
import torch
from torchvision import transforms
from merge_utils import merge_attribute_value, merge_attribute_list
from grouping_map import MERGE_MAP
from dataset.kfashion_dataset_all import ATTRIBUTE_LOOKUP, ATTRIBUTE_SCHEMA
from dataset.style_dataset import STYLE_CATEGORIES, SUBSTYLE_CATEGORIES
import requests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(raw_bytes):
    image = Image.open(raw_bytes).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def run_inference(image_tensor, type_model, attr_model, style_model):
    # 1단계: 의류 종류 분류
    with torch.no_grad():
        clothing_idx = type_model(image_tensor).argmax(1).item()
        clothing_label = ["상의", "하의", "아우터", "원피스"][clothing_idx]

    # 2단계: 속성 예측
    with torch.no_grad():
        attr_outputs = attr_model(image_tensor)

    # 3단계: 스타일/서브스타일 예측
    with torch.no_grad():
        style_outputs = style_model(image_tensor)
        style_idx = torch.argmax(style_outputs["style"], 1).item()
        substyle_idx = torch.argmax(style_outputs["substyle"], 1).item()
        style_name = STYLE_CATEGORIES[style_idx]
        substyle_name = SUBSTYLE_CATEGORIES[substyle_idx]

    # 스타일 병합
    for group, items in MERGE_MAP["스타일"].items():
        if style_name in items:
            style_name = group
        if substyle_name in items:
            substyle_name = group

    # 결과 초기화
    result = {
        "의류종류": clothing_label,
        "스타일": {
            "스타일": style_name,
            "서브스타일": substyle_name
        },
        "속성": {part: {} for part in ATTRIBUTE_SCHEMA}
    }

    for attr in attr_outputs:
        part, attr_name = attr.split("_")
        if part != clothing_label:
            continue

        logits = attr_outputs[attr].squeeze(0)

        if attr_name in ATTRIBUTE_SCHEMA[part]["single"]:
            # 단일 속성 예측
            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(probs[1:]) + 1  # skip background (0)
            decoded = ATTRIBUTE_LOOKUP[part][attr_name][pred - 1]
            merged = merge_attribute_value(attr_name, decoded)
            result["속성"][part][attr_name] = merged

        else:
            # 다중 속성 예측
            probs = torch.sigmoid(logits)
            topk = torch.topk(probs[1:], 3)
            indices = topk.indices + 1  # skip background
            values = [ATTRIBUTE_LOOKUP[part][attr_name][i - 1] for i in indices if probs[i] > 0.5]

            if not values:
                top_idx = torch.argmax(probs[1:]) + 1
                values = [ATTRIBUTE_LOOKUP[part][attr_name][top_idx - 1]]

            merged_values = merge_attribute_list(attr_name, values)
            result["속성"][part][attr_name] = merged_values

    return result

def send_result_to_spring(clothing_json, image_filename):
    url = "http://localhost:8080/api/clothing"  # Spring API endpoint
    data = clothing_json
    data['imagePath'] = f"/images/{image_filename}"  # 로컬 저장 경로 or URL

    response = requests.post(url, json=data)
    print("Spring 응답:", response.status_code, response.text)
    return response