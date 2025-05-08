from fastapi import FastAPI, UploadFile, File
from model_loader import load_models
from inference import preprocess_image, run_inference
from io import BytesIO
from PIL import Image
import uvicorn
import os
import shutil
import uuid
from cropModel.crop import apply_grabcut, rembg, cv2
from bodyModel.body_shape_detector import detect_body_shape_from_bytes

app = FastAPI()

type_model, attr_model, style_model = load_models()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    ext = file.filename.split(".")[-1]
    unique_id = str(uuid.uuid4())
    saved_filename = f"{unique_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(file_path, "rb") as f:
        image_tensor = preprocess_image(BytesIO(f.read()))
    result = run_inference(image_tensor, type_model, attr_model, style_model)

    original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    rembg_removed = rembg.remove(original_img)
    cropped_img = apply_grabcut(rembg_removed)
    processed_path = os.path.join(PROCESSED_DIR, f"{unique_id}_processed.png")
    cv2.imwrite(processed_path, cropped_img)

    result["imageUrl"] = f"/{processed_path.replace(os.sep, '/')}"

    return result

@app.post("/body-shape")
async def body_shape(file: UploadFile = File(...)):
    image_bytes = await file.read()
    shape = detect_body_shape_from_bytes(image_bytes)
    return [shape]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
