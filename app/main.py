# app/main.py
import io
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .model import LightningBirdsClassifier
from .inference import classify
from PIL import Image
import torch
import onnx


app = FastAPI()

MODEL_PATH = "./app/birds_model.pt"
ONNX_PATH = "./app/birds_model.onnx"
model = LightningBirdsClassifier.load_from_checkpoint(
    MODEL_PATH,
    map_location=torch.device('cpu'),
    lr=1e-4,
    transfer=False)

class InputData(BaseModel):
    feature1: float
    feature2: float
    ...
    
@app.get("/", summary="Root")
async def root():
    return {"message": "Birds Classification API"}

@app.post("/forward")
async def forward(image: UploadFile = File(None)):
    if image is not None:
        try:
            img_bytes = await image.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(400, "bad request")
        try:
            pred_label = classify(model, pil_img)
        except Exception as e:
            raise HTTPException(403, "модель не смогла обработать данные")

        return JSONResponse(content={"predicted_label": pred_label})
    
    return HTTPException(403, "error")

@app.get("/metadata")
async def metadata():
    model = onnx.load(ONNX_PATH)
    meta = {p.key: p.value for p in model.metadata_props}
    return JSONResponse(content=meta)

