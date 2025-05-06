from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch
import glob

# -----------------------------
# 의류별 속성 정의
# -----------------------------
ATTRIBUTE_SCHEMA = {
    "상의": {
        "single": ["카테고리", "기장", "소매기장", "넥라인", "칼라", "핏"],
        "multi": ["색상", "소재", "디테일", "프린트"]
    },
    "하의": {
        "single": ["카테고리", "기장", "핏"],
        "multi": ["색상", "소재", "디테일", "프린트"]
    },
    "아우터": {
        "single": ["카테고리", "기장", "소매기장", "넥라인", "칼라", "핏"],
        "multi": ["색상", "소재", "디테일", "프린트"]
    },
    "원피스": {
        "single": ["카테고리", "기장", "소매기장", "넥라인", "칼라", "핏"],
        "multi": ["색상", "소재", "디테일", "프린트"]
    }
}

# -----------------------------
# Value 리스트 (예시값으로 구성, 실제와 다를 수 있음)
# -----------------------------
COLOR_CATEGORIES = ["블랙", "화이트", "그레이", "레드", "핑크", "오렌지", "베이지", "브라운", "옐로우", "그린", "카키", "민트", "블루", "네이비", "스카이블루", "퍼플", "라벤더", "와인", "네온", "골드", "실버"]
MATERIAL_CATEGORIES = ["패딩", "무스탕", "퍼프", "네오프렌", "코듀로이", "트위드", "자카드", "니트", "페플럼", "레이스", "스판덱스", "메시", "비닐/PVC", "데님", "울/캐시미어", "저지", "시퀸/글리터", "퍼", "헤어 니트", "실크", "린넨", "플리스", "시폰", "스웨이드", "가죽", "우븐", "벨벳"]
DETAIL_CATEGORIES = ["스터드", "드롭숄더", "드롭웨이스트", "레이스업", "슬릿", "프릴", "단추", "퀄팅", "스팽글", "롤업", "니트꽈베기", "체인", "프린지", "지퍼", "태슬", "띠", "플레어", "싱글브레스티드", "더블브레스티드", "스트링", "자수", "폼폼", "디스트로이드", "페플럼", "X스트랩", "스티치", "레이스", "퍼프", "비즈", "컷아웃", "버클", "포켓", "러플", "글리터", "퍼트리밍", "플리츠", "비대칭", "셔링", "패치워크", "리본"]
PRINT_CATEGORIES = ["페이즐리", "하트", "지그재그", "깅엄", "하운즈 투스", "도트", "레터링", "믹스", "뱀피", "해골", "체크", "무지", "카무플라쥬", "그라데이션", "스트라이프", "호피", "아가일", "그래픽", "지브라", "타이다이", "플로럴"]

SLEEVE_CATEGORIES = ["민소매", "반팔", "긴팔", "7부소매", "캡"]
NECKLINE_CATEGORIES = ["라운드넥", "브이넥", "유넥", "홀터넥", "원 숄더", "오프숄더", "스퀘어넥", "노카라", "후드", "터틀넥", "보트넥", "스위트하트"]
COLLAR_CATEGORIES = ["세일러칼라", "셔츠칼라", "보우칼라", "차이나칼라", "숄칼라", "폴로칼라", "피터팬칼라", "너치드칼라","테일러드칼라", "밴드칼라"]
TOP_FIT_CATEGORIES = ["루즈", "슬림", "레귤러", "오버사이즈", "타이트"]
BOTTOM_FIT_CATEGORIES = ["스키니", "노멀", "와이드", "루즈", "벨보텀"]
OUTER_FIT_CATEGORIES = ["루즈", "슬림", "레귤러", "오버사이즈", "타이트"]
DRESS_FIT_CATEGORIES = ["루즈", "슬림", "레귤러", "오버사이즈", "타이트"]

TOP_CATEGORIES = ["탑", "블라우스", "티셔츠", "니트웨어", "셔츠", "브라탑", "후드티"]
BOTTOM_CATEGORIES = ["청바지", "팬츠", "스커트", "레깅스", "조거팬츠"]
OUTER_CATEGORIES = ["코트", "재킷", "점퍼", "패딩", "베스트", "가디건", "짚업"]
DRESS_CATEGORIES = ["드레스", "점프수트"]

TOP_LENGTH_CATEGORIES = ["크롭", "노멀", "롱"]
BOTTOM_LENGTH_CATEGORIES = ["미니", "니렝스", "미디", "발목", "맥시"]
OUTER_LENGTH_CATEGORIES = ["크롭", "노멀", "하프", "롱", "맥시"]
DRESS_LENGTH_CATEGORIES = ["미니", "니렝스", "미디", "발목", "맥시"]

# -----------------------------
# LOOKUP 테이블
# -----------------------------
def build_lookup():
    return {
        "상의": {
            "카테고리": TOP_CATEGORIES,
            "기장": TOP_LENGTH_CATEGORIES,
            "소매기장": SLEEVE_CATEGORIES,
            "넥라인": NECKLINE_CATEGORIES,
            "칼라": COLLAR_CATEGORIES,
            "핏": TOP_FIT_CATEGORIES,
            "색상": COLOR_CATEGORIES,
            "소재": MATERIAL_CATEGORIES,
            "디테일": DETAIL_CATEGORIES,
            "프린트": PRINT_CATEGORIES
        },
        "하의": {
            "카테고리": BOTTOM_CATEGORIES,
            "기장": BOTTOM_LENGTH_CATEGORIES,
            "핏": BOTTOM_FIT_CATEGORIES,
            "색상": COLOR_CATEGORIES,
            "소재": MATERIAL_CATEGORIES,
            "디테일": DETAIL_CATEGORIES,
            "프린트": PRINT_CATEGORIES
        },
        "아우터": {
            "카테고리": OUTER_CATEGORIES,
            "기장": OUTER_LENGTH_CATEGORIES,
            "소매기장": SLEEVE_CATEGORIES,
            "넥라인": NECKLINE_CATEGORIES,
            "칼라": COLLAR_CATEGORIES,
            "핏": OUTER_FIT_CATEGORIES,
            "색상": COLOR_CATEGORIES,
            "소재": MATERIAL_CATEGORIES,
            "디테일": DETAIL_CATEGORIES,
            "프린트": PRINT_CATEGORIES
        },
        "원피스": {
            "카테고리": DRESS_CATEGORIES,
            "기장": DRESS_LENGTH_CATEGORIES,
            "소매기장": SLEEVE_CATEGORIES,
            "넥라인": NECKLINE_CATEGORIES,
            "칼라": COLLAR_CATEGORIES,
            "핏": DRESS_FIT_CATEGORIES,
            "색상": COLOR_CATEGORIES,
            "소재": MATERIAL_CATEGORIES,
            "디테일": DETAIL_CATEGORIES,
            "프린트": PRINT_CATEGORIES
        }
    }

ATTRIBUTE_LOOKUP = build_lookup()



# -----------------------------
# Dataset 클래스
# -----------------------------
class KFashionDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
        for json_path in json_files:
            fname = os.path.splitext(os.path.basename(json_path))[0]
            relative = os.path.relpath(os.path.dirname(json_path), json_dir)
            img_path = os.path.join(image_dir, relative, fname + ".jpg")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.label_paths.append(json_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(self.label_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)

        label_root = data.get("데이터셋 정보", {}).get("데이터셋 상세설명", {}).get("라벨링", {})
        encoded = {}

        for part in ATTRIBUTE_SCHEMA:
            if part not in label_root or not label_root[part] or not isinstance(label_root[part], list):
                continue

            attr_data = label_root[part][0]
            if not attr_data:
                continue

            # single-label
            for attr in ATTRIBUTE_SCHEMA[part]["single"]:
                value = attr_data.get(attr)
                lookup = ATTRIBUTE_LOOKUP[part][attr]
                key = f"{part}_{attr}"
                encoded[key] = lookup.index(value) + 1 if value in lookup else 0

            # multi-label
            for attr in ATTRIBUTE_SCHEMA[part]["multi"]:
                values = attr_data.get(attr, [])
                if not isinstance(values, list):
                    values = [values]
                lookup = ATTRIBUTE_LOOKUP[part][attr]
                onehot = [0] * (len(lookup) + 1)
                if not values:
                    onehot[0] = 1
                else:
                    for v in values:
                        if v in lookup:
                            onehot[lookup.index(v) + 1] = 1
                        else:
                            onehot[0] = 1
                key = f"{part}_{attr}"
                encoded[key] = onehot

        return image, encoded