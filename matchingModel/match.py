import requests, os, numpy as np, torch
from PIL import Image
from ultralyticsplus import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import clip

clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL_PATH = "kesimeg/yolov8n-clothing-detection"

def match_image_against_db(user_id, image_path):
    closet_url = f"http://localhost:8080/api/internal/closet/{user_id}"
    closet_res = requests.get(closet_url)
    closet_data = closet_res.json()

    outfit_parts = detect_outfit_parts(image_path)
    part_images = [p[0] for p in outfit_parts]

    matched = []
    for part_img in part_images:
        tensor = preprocess(part_img).unsqueeze(0).to(clip_model.visual.device)
        with torch.no_grad():
            part_feat = clip_model.encode_image(tensor)
            part_feat /= part_feat.norm(dim=-1, keepdim=True)
        part_feat = part_feat.cpu().numpy()

        best_item = None
        best_score = -1
        for cloth in closet_data:
            cloth_path = f"./uploads/original/{cloth['imagePath'].split('/')[-1]}"
            if not os.path.exists(cloth_path): continue

            img = Image.open(cloth_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(clip_model.visual.device)
            with torch.no_grad():
                emb = clip_model.encode_image(img_tensor)
                emb /= emb.norm(dim=-1, keepdim=True)

            sim = cosine_similarity(part_feat, emb.cpu().numpy())[0][0]
            if sim > best_score:
                best_score = sim
                best_item = cloth

        if best_item:
            matched.append({
                "category": best_item.get("category"),
                "clothId": best_item["clothId"],
                "imagePath": best_item["imagePath"],
                "croppedPath": best_item["croppedPath"],
                "score": round(float(best_score), 4)
            })

    return matched

def detect_outfit_parts(image_path):
    model = YOLO(YOLO_MODEL_PATH)
    results = model(image_path)
    image = Image.open(image_path).convert("RGB")
    parts = []
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    labels = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names

    for box, cls in zip(boxes, labels):
        cropped = image.crop(box)
        label = names[cls]
        parts.append((cropped, label))
    return parts
