from fastapi import FastAPI, UploadFile, File
from model_loader import load_models
from inference import preprocess_image, run_inference
from io import BytesIO
import uvicorn

app = FastAPI()
type_model, attr_model, style_model = load_models()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_tensor = preprocess_image(BytesIO(await file.read()))
    result = run_inference(image_tensor, type_model, attr_model, style_model)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)