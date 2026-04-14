# ============================================================
# app/api.py — FastAPI REST API
# AI-Powered Medical Image Analysis System
# ============================================================
#
# HOW TO RUN:
#   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
#
# ENDPOINTS:
#   GET  /           → Health check
#   GET  /info       → Model info
#   POST /predict    → Upload image → get prediction JSON
#   POST /predict/gradcam → Prediction + Grad-CAM image
#
# EXAMPLE:
#   curl -X POST http://localhost:8000/predict \
#        -F "file=@chest_xray.jpg"
#
# RESPONSE:
#   {
#     "prediction": "PNEUMONIA",
#     "confidence": 0.93,
#     "probabilities": {"NORMAL": 0.07, "PNEUMONIA": 0.93},
#     "model": "MobileNetV2",
#     "status": "success"
#   }
# ============================================================

import os
import sys
import io
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import load_trained_model
from src.preprocess import preprocess_pil_image
from src.evaluate import predict_single
from src.gradcam import compute_gradcam, overlay_gradcam, _find_last_conv_layer
from src.utils import load_config

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="AI Medical Image Analysis API",
    description=(
        "REST API for chest X-ray disease classification "
        "using MobileNetV2 transfer learning and Grad-CAM explainability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS (allows frontend to call this API) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Config & Model at Startup ────────────────────────────
cfg = load_config()
CLASS_LABELS = cfg["dataset"]["classes"]
IMG_SIZE     = tuple(cfg["dataset"]["image_size"])
MODEL_PATH   = os.path.join(cfg["paths"]["models"], "best_model.keras")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Please train first: python main.py --mode train"
            )
        _model = load_trained_model(MODEL_PATH)
    return _model


# ── Pydantic Response Models ──────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: dict
    model:         str
    timestamp:     str
    status:        str


class HealthResponse(BaseModel):
    status:     str
    model_ready: bool
    version:    str
    timestamp:  str


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint — returns simple HTML welcome page."""
    return """
    <html>
    <head><title>AI Medical API</title></head>
    <body style="font-family:Arial; max-width:600px; margin:50px auto; text-align:center">
        <h1>🏥 AI Medical Image Analysis API</h1>
        <p>Version 2.0.0 | Powered by MobileNetV2 + TensorFlow</p>
        <hr>
        <p>
          <a href="/docs">📖 Swagger Docs</a> &nbsp;|&nbsp;
          <a href="/redoc">📚 ReDoc</a> &nbsp;|&nbsp;
          <a href="/health">🩺 Health Check</a>
        </p>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        model = get_model()
        model_ready = True
    except Exception:
        model_ready = False

    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        model_ready=model_ready,
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/info")
async def model_info():
    """Returns model and system information."""
    return {
        "model":        cfg["model"]["backbone"],
        "num_classes":  cfg["dataset"]["num_classes"],
        "classes":      CLASS_LABELS,
        "image_size":   IMG_SIZE,
        "dataset":      cfg["dataset"]["name"],
        "explainability": "Grad-CAM",
        "version":      "2.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.

    - **file**: Upload a chest X-ray image (JPG/PNG)

    Returns:
    - prediction: predicted class name
    - confidence: confidence score (0–1)
    - probabilities: per-class probability dict
    """
    # ── Validate file type ────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPG or PNG."
        )

    try:
        # ── Read & preprocess ─────────────────────────────
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess_pil_image(pil_image, IMG_SIZE)

        # ── Predict ───────────────────────────────────────
        model = get_model()
        result = predict_single(model, img_array, CLASS_LABELS)

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model=cfg["model"]["backbone"],
            timestamp=datetime.now().isoformat(),
            status="success"
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Prediction + Grad-CAM endpoint.
    Returns prediction JSON AND base64-encoded Grad-CAM overlay image.

    The base64 image can be displayed in any frontend with:
      <img src="data:image/png;base64,{gradcam_b64}" />
    """
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess_pil_image(pil_image, IMG_SIZE)

        model = get_model()
        result = predict_single(model, img_array, CLASS_LABELS)

        # Grad-CAM
        pred_idx  = CLASS_LABELS.index(result["prediction"])
        last_conv = _find_last_conv_layer(model)
        heatmap   = compute_gradcam(model, img_array,
                                     pred_index=pred_idx,
                                     last_conv_layer_name=last_conv)
        overlaid  = overlay_gradcam(img_array[0], heatmap)

        # Encode to base64
        overlaid_pil = Image.fromarray(overlaid)
        buffer = io.BytesIO()
        overlaid_pil.save(buffer, format="PNG")
        gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "prediction":    result["prediction"],
            "confidence":    result["confidence"],
            "probabilities": result["probabilities"],
            "gradcam_image": gradcam_b64,   # base64 PNG
            "model":         cfg["model"]["backbone"],
            "timestamp":     datetime.now().isoformat(),
            "status":        "success"
        })

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=True
    )
