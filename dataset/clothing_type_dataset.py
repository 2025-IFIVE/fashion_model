import os
import glob
import json
from PIL import Image
from torch.utils.data import Dataset

# 라벨 정의
LABELS = ["상의", "하의", "아우터", "원피스"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

class ClothingTypeDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
        for json_path in json_files:
            fname = os.path.splitext(os.path.basename(json_path))[0]
            rel = os.path.relpath(os.path.dirname(json_path), json_dir)
            img_path = os.path.join(image_dir, rel, fname + ".jpg")
            if not os.path.exists(img_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            label_data = data.get("데이터셋 정보", {}).get("데이터셋 상세설명", {}).get("라벨링", {})
            for part in LABELS:
                if part in label_data and isinstance(label_data[part], list) and label_data[part] and label_data[part][0] != {}:
                    self.image_paths.append(img_path)
                    self.labels.append(LABEL_TO_IDX[part])
                    break  # 하나만 존재한다고 가정

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label