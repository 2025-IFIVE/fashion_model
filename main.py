from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_loader import load_models
from inference import preprocess_image, run_inference

from cropModel.crop import apply_grabcut, cv2  
from bodyModel.body_shape_detector import detect_body_shape_from_bytes
from bodyModel.body_shape_pose import detect_body_shape_with_pose_and_segmentation
from recommendModel.recommend_logic import generate_recommendation
from matchingModel.match import match_image_against_db

from PIL import Image, ImageOps
from io import BytesIO
import io
import numpy as np
import uvicorn
import os
import shutil
import uuid
import requests
import jwt

# ─────────────────────────────────────────────────────────
# FastAPI & CORS
# ─────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Config / Models
# ─────────────────────────────────────────────────────────
type_model, attr_model, style_model = load_models()

SPRING_BASE_URL   = os.getenv("SPRING_BASE_URL", "http://localhost:8080")
SPRING_UPLOAD_URL = os.getenv("SPRING_UPLOAD_URL", f"{SPRING_BASE_URL}/api/internal/upload-cropped")
JWT_SECRET        = os.getenv("JWT_SECRET", "mysupersecretkeythatshouldbelongenough")
JWT_ALGORITHM     = os.getenv("JWT_ALGORITHM", "HS256")

UPLOAD_DIR    = "uploads"
ORIGINAL_DIR  = os.path.join(UPLOAD_DIR, "original")
PROCESSED_DIR = "processed"
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def send_cropped_to_spring(cropped_path: str, filename: str) -> str:
    with open(cropped_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        response = requests.post(SPRING_UPLOAD_URL, files=files, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Spring 업로드 실패: {response.status_code} - {response.text}")
        return response.text.strip()

def extract_user_id_from_token(auth_header: str) -> int:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise Exception("Authorization 헤더가 없거나 잘못되었습니다.")
    token = auth_header.replace("Bearer ", "")
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    uid = payload.get("userId")
    if uid is None:
        raise Exception("JWT payload에 userId가 없습니다.")
    return int(uid)

def decode_upload_to_bgr(data: bytes) -> np.ndarray:
    """
    업로드 바이트 → EXIF 회전 보정 → RGBA → BGRA → BGR 일관화
    (알파/배경제거는 모델 내부에서만 수행)
    """
    pil = Image.open(io.BytesIO(data))
    pil = ImageOps.exif_transpose(pil).convert("RGBA")   # 회전 반영 + RGBA
    rgba = np.array(pil)                                  # (H, W, 4) RGBA
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)       # RGBA -> BGRA
    bgr  = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)        # BGRA -> BGR
    return bgr

# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 파일명/경로 준비
    ext = (file.filename.split(".")[-1] or "jpg").lower()
    unique_id = str(uuid.uuid4())
    saved_filename = f"{unique_id}.{ext}"
    file_path = os.path.join(ORIGINAL_DIR, saved_filename)

    # 1) 원본 저장
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 추론 입력 준비 (분류/속성/스타일)
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    image_tensor = preprocess_image(BytesIO(raw_bytes))
    result = run_inference(image_tensor, type_model, attr_model, style_model)

    # 3) 배경제거/의류 크롭: 메인에서는 rembg 금지, 원본 BGR만 디코드
    img_bgr = decode_upload_to_bgr(raw_bytes)  # EXIF 회전 & BGR 일관화
    cropped_img = apply_grabcut(img_bgr)       # 내부에서 RMBG-1.4 + CLIPSeg 사용 (BGRA 반환)

    # 4) 저장 & 스프링 업로드
    processed_filename = f"{unique_id}_processed.png"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    cv2.imwrite(processed_path, cropped_img)   # BGRA PNG 저장

    cropped_url = send_cropped_to_spring(processed_path, processed_filename)

    # 5) 응답 (기존 스키마 유지)
    result["imageUrl"] = f"/uploads/original/{saved_filename}"
    result["croppedUrl"] = cropped_url
    return result

@app.post("/body-shape")
async def body_shape(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return ["이미지 디코딩 실패"]
    shape, _ = detect_body_shape_with_pose_and_segmentation(image)
    return [shape]

class RecommendRequestDTO(BaseModel):
    userId: int
    weather: str  # 예: "23~27도"

@app.post("/recommend")
def recommend(req: RecommendRequestDTO):
    try:
        spring_url = f"{SPRING_BASE_URL}/api/clothing/user/{req.userId}"
        response = requests.get(spring_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Spring 통신 실패: {response.status_code}")

        closet_items = response.json()  # [{...}]
        result = generate_recommendation(closet_items, req.weather)
        if result is None:
            raise HTTPException(status_code=404, detail="추천 가능한 조합이 없습니다.")

        return {
            "userId": req.userId,
            "weather": req.weather,
            "recommendation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match")
async def match(file: UploadFile = File(...), authorization: str = Header(...)):
    print("/match 요청 수신")
    user_id = extract_user_id_from_token(authorization)
    print("추출된 userId:", user_id)

    ext = (file.filename.split(".")[-1] or "jpg").lower()
    file_name = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(ORIGINAL_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print("저장된 이미지 경로:", save_path)

    try:
        match_result = match_image_against_db(user_id=user_id, image_path=save_path)
        print("매칭 결과:", match_result)
    except Exception as e:
        print("매칭 처리 중 오류:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # 기존 프런트/스프링과 호환되는 응답 형식
    seen_ids = set()
    matchedImages, clothIds, labels, scores = [], [], [], []
    for m in match_result.get("matches", []):
        cid = m["matchedClothId"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        matchedImages.append(m["matchedImagePath"])
        clothIds.append(cid)
        labels.append(m["partLabel"])
        scores.append(m["similarity"])

    return {"matchedImages": matchedImages, "clothIds": clothIds, "labels": labels, "scores": scores}

# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 필요 시 호스트/포트 환경변수로 조절 가능
    uvicorn.run(app, host="0.0.0.0", port=8000)
