#!/usr/bin/env python3
"""FastAPI server for AI-human text detection."""

import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multilingual_ai_detection.models import load_model

app = FastAPI(
    title="Multilingual AI-Human Text Detection API",
    description="API for detecting AI-generated vs human-written text across multiple languages",
    version="0.1.0"
)

# Global model variable
model = None
model_type = None

class PredictionRequest(BaseModel):
    text: str
    return_probabilities: Optional[bool] = False

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    return_probabilities: Optional[bool] = False

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Optional[dict] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.on_event("startup")
async def load_model_on_startup():
    """Load model on startup."""
    global model, model_type

    model_path = os.getenv("MODEL_PATH", "models/statistical")
    model_type_env = os.getenv("MODEL_TYPE", "statistical")

    if not Path(model_path).exists():
        print(f"Warning: Model path {model_path} does not exist")
        return

    try:
        print(f"Loading {model_type_env} model from {model_path}...")
        model = load_model(model_path, model_type_env)
        model_type = model_type_env
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multilingual AI-Human Text Detection API",
        "model_loaded": model is not None,
        "model_type": model_type
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict if text is AI-generated or human-written."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = model.predict([request.text])
        probabilities = model.predict_proba([request.text])

        class_names = ["Human", "AI"]
        prediction = class_names[predictions[0]]
        confidence = max(probabilities[0]) if isinstance(probabilities[0], list) else probabilities[0]

        response = PredictionResponse(
            text=request.text,
            prediction=prediction,
            confidence=confidence
        )

        if request.return_probabilities:
            if isinstance(probabilities[0], list):
                response.probabilities = {
                    class_names[i]: prob for i, prob in enumerate(probabilities[0])
                }
            else:
                response.probabilities = {"score": probabilities[0]}

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")

    try:
        predictions = model.predict(request.texts)
        probabilities = model.predict_proba(request.texts)

        class_names = ["Human", "AI"]
        results = []

        for text, pred, prob in zip(request.texts, predictions, probabilities):
            prediction = class_names[pred]
            confidence = max(prob) if isinstance(prob, list) else prob

            response = PredictionResponse(
                text=text,
                prediction=prediction,
                confidence=confidence
            )

            if request.return_probabilities:
                if isinstance(prob, list):
                    response.probabilities = {
                        class_names[i]: p for i, p in enumerate(prob)
                    }
                else:
                    response.probabilities = {"score": prob}

            results.append(response)

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port, reload=False)