from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from model_loader import load_models
from inference import preprocess_image, run_inference
from io import BytesIO
import numpy as np
import uvicorn
import os
import shutil
import uuid
import requests
import jwt

from cropModel.crop import apply_grabcut, rembg, cv2
from bodyModel.body_shape_detector import detect_body_shape_from_bytes
from bodyModel.body_shape_pose import detect_body_shape_with_pose_and_segmentation
from recommendModel.recommend_logic import generate_recommendation
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from matchingModel.match import match_image_against_db  

from matchingModel.match_yolo import match_image_against_db_v2


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

type_model, attr_model, style_model = load_models()

UPLOAD_DIR = "uploads"
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
PROCESSED_DIR = "processed"
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SPRING_UPLOAD_URL = "http://localhost:8080/api/internal/upload-cropped"
JWT_SECRET = "mysupersecretkeythatshouldbelongenough"
JWT_ALGORITHM = "HS256"

def send_cropped_to_spring(cropped_path: str, filename: str):
    with open(cropped_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        response = requests.post(SPRING_UPLOAD_URL, files=files)
        if response.status_code != 200:
            raise Exception(f"Spring 업로드 실패: {response.status_code} - {response.text}")
        return response.text.strip()

# JWT 토큰에서 userId 추출
def extract_user_id_from_token(auth_header: str) -> int:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise Exception("Authorization 헤더가 없거나 잘못되었습니다.")
    token = auth_header.replace("Bearer ", "")
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload.get("userId")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    unique_id = str(uuid.uuid4())
    saved_filename = f"{unique_id}.{ext}"
    file_path = os.path.join(ORIGINAL_DIR, saved_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(file_path, "rb") as f:
        image_tensor = preprocess_image(BytesIO(f.read()))
    result = run_inference(image_tensor, type_model, attr_model, style_model)

    original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    rembg_removed = rembg.remove(original_img)
    cropped_img = apply_grabcut(rembg_removed)

    processed_filename = f"{unique_id}_processed.png"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    cv2.imwrite(processed_path, cropped_img)

    cropped_url = send_cropped_to_spring(processed_path, processed_filename)

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
    shape = detect_body_shape_from_bytes(image_bytes)
    return {"bodyShape": shape}

# 요청 바디 스키마
class RecommendRequestDTO(BaseModel):
    userId: int
    weather: str  # 예: "23~27도"

    # 추천 요청 엔드포인트
@app.post("/recommend")
def recommend(req: RecommendRequestDTO):
    try:
        # Spring 서버에서 유저 옷 데이터 가져오기
        spring_url = f"http://localhost:8080/api/clothing/user/{req.userId}"
        response = requests.get(spring_url)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Spring 통신 실패: {response.status_code}")

        closet_items = response.json()  # 예상 구조: [{ "id": ..., "type": "상의", "imagePath": ... }, ...]

        # 추천 생성
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

'''
# 이미지 매칭 API 추가
@app.post("/match")
async def match(file: UploadFile = File(...), authorization: str = Header(...)):
    print("/match 요청 수신")
    try:
        user_id = extract_user_id_from_token(authorization)
        print("추출된 userId:", user_id)
    except Exception as e:
        print("JWT 오류:", str(e))
        raise

    ext = file.filename.split(".")[-1]
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
        raise

    return match_result
    '''
#매칭
@app.post("/match")
async def match(file: UploadFile = File(...), authorization: str = Header(...)):
    print("/match 요청 수신")
    try:
        user_id = extract_user_id_from_token(authorization)
        print("추출된 userId:", user_id)
    except Exception as e:
        print("JWT 오류:", str(e))
        raise

    ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(ORIGINAL_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print("저장된 이미지 경로:", save_path)

    try:
        # ✅ CLIP → YOLO+속성 기반으로 변경
        match_result = match_image_against_db_v2(user_id=user_id, image_path=save_path)
        print("매칭 결과:", match_result)
    except Exception as e:
        print("매칭 처리 중 오류:", str(e))
        raise

    return match_result
