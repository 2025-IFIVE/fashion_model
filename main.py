from fastapi import FastAPI, UploadFile, File
from model_loader import load_models
from inference import preprocess_image, run_inference
from io import BytesIO
import numpy as np
import uvicorn
import os
import shutil
import uuid
import requests
from cropModel.crop import apply_grabcut, rembg, cv2
from bodyModel.body_shape_detector import detect_body_shape_from_bytes
from bodyModel.body_shape_pose import detect_body_shape_with_pose_and_segmentation

app = FastAPI()

type_model, attr_model, style_model = load_models()

UPLOAD_DIR = "uploads"
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
PROCESSED_DIR = "processed"
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SPRING_UPLOAD_URL = "http://localhost:8080/api/internal/upload-cropped"

def send_cropped_to_spring(cropped_path: str, filename: str):
    with open(cropped_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        response = requests.post(SPRING_UPLOAD_URL, files=files)
        if response.status_code != 200:
            raise Exception(f"Spring 업로드 실패: {response.status_code} - {response.text}")
        return response.text.strip()

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