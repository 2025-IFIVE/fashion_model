import os
import glob
import json
from PIL import Image
from torch.utils.data import Dataset

# 스타일 카테고리 정의 (예시)

STYLE_CATEGORIES = ["기타", "레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티", "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피", "히피", "힙합" ]  # 예시
SUBSTYLE_CATEGORIES = ["기타", "레트로", "로맨틱", "리조트", "매니시", "모던", "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티", "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리", "클래식", "키치", "톰보이", "펑크", "페미닌", "프레피", "히피", "힙합" ]  # 예시

STYLE_TO_IDX = {s: i for i, s in enumerate(STYLE_CATEGORIES)}
SUBSTYLE_TO_IDX = {s: i for i, s in enumerate(SUBSTYLE_CATEGORIES)}

class StyleDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_paths = []
        self.styles = []
        self.substyles = []
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
            style_info = label_data.get("스타일", [{}])[0]
            style = style_info.get("스타일")
            substyle = style_info.get("서브스타일")

            if style not in STYLE_TO_IDX or substyle not in SUBSTYLE_TO_IDX:
                continue

            self.image_paths.append(img_path)
            self.styles.append(STYLE_TO_IDX[style])
            self.substyles.append(SUBSTYLE_TO_IDX[substyle])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.styles[idx], self.substyles[idx]