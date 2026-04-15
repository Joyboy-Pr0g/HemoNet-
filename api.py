"""
HemoNet — FastAPI Backend (Step h)

Run:
  py -m pip install -r requirements.txt
  py -m uvicorn api:app --reload --port 8000
"""

import io
import numpy as np
from pathlib import Path
from typing import List

import torch
import pydicom
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import CustomCNN, build_resnet50
from dataset_preparation import build_val_test_transforms, IMG_SIZE

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
THRESHOLD          = 0.5
RESNET_PATH        = "saved_models/ResNet50.pth"
CUSTOM_PATH        = "saved_models/CustomCNN.pth"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".dcm"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[API] Device: {device}")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("[API] Loading ResNet50...")
resnet_model = build_resnet50().to(device)
resnet_model.load_state_dict(
    torch.load(RESNET_PATH, map_location=device))
resnet_model.eval()

print("[API] Loading CustomCNN...")
custom_model = CustomCNN().to(device)
custom_model.load_state_dict(
    torch.load(CUSTOM_PATH, map_location=device))
custom_model.eval()

print("[API] Both models ready.\n")

transform = build_val_test_transforms(IMG_SIZE)


def load_image(file_bytes: bytes, filename: str) -> Image.Image:
    ext = Path(filename).suffix.lower()

    if ext == ".dcm":
        dicom     = pydicom.dcmread(io.BytesIO(file_bytes))
        pixels    = dicom.pixel_array.astype(np.float32)

        slope     = float(getattr(dicom, "RescaleSlope",     1))
        intercept = float(getattr(dicom, "RescaleIntercept", 0))
        pixels    = pixels * slope + intercept

        # Standard brain CT window: WL=40, WW=80
        wl, ww = 40, 80
        lower, upper = wl - ww / 2, wl + ww / 2
        pixels = np.clip(pixels, lower, upper)
        pixels = ((pixels - lower) / (upper - lower) * 255).astype(np.uint8)

        return Image.fromarray(pixels).convert("RGB")

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def predict_single(model, image: Image.Image) -> dict:
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(tensor).squeeze().item()

    prediction = "Hemorrhage" if prob >= THRESHOLD else "Normal"
    confidence = round(abs(prob - 0.5) * 2, 4)

    return {
        "prediction" : prediction,
        "probability": round(prob, 4),
        "confidence" : confidence,
    }


app = FastAPI(title="HemoNet API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "running", "models": ["ResNet50", "CustomCNN"]}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 16:
        raise HTTPException(status_code=400,
                            detail="Maximum 16 images per request.")

    results = []

    for upload in files:
        filename = upload.filename or "unknown"
        ext      = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}")

        try:
            image = load_image(await upload.read(), filename)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{filename}': {str(e)}")

        r = predict_single(resnet_model, image)
        c = predict_single(custom_model, image)

        print(f"  [{filename}] ResNet50={r['prediction']} ({r['probability']:.3f}) "
              f"| CustomCNN={c['prediction']} ({c['probability']:.3f})")

        results.append({
            "filename" : filename,
            "resnet50" : r,
            "customcnn": c,
        })

    hemo_r    = sum(1 for r in results if r["resnet50"]["prediction"]  == "Hemorrhage")
    hemo_c    = sum(1 for r in results if r["customcnn"]["prediction"] == "Hemorrhage")
    agreement = sum(1 for r in results if r["resnet50"]["prediction"]  == r["customcnn"]["prediction"])

    return {
        "results": results,
        "summary": {
            "total"            : len(results),
            "hemorrhage_resnet": hemo_r,
            "hemorrhage_custom": hemo_c,
            "agreement"        : agreement,
            "disagreement"     : len(results) - agreement,
        }
    }