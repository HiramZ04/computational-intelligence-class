from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

# --------- Schemas ---------
class IrisFeatures(BaseModel):
    sepal_lenght: float = Field(..., ge=0, description="Longitud del sepalo")
    sepal_width: float  = Field(..., ge=0, description="Ancho del sepalo")
    petal_lenght: float = Field(..., ge=0, description="Longitud del petalo")
    petal_width: float  = Field(..., ge=0, description="Ancho del petalo")

class PredictResponse(BaseModel):
    prediction: int
    species: str
    confidence: float   # <-- era str, lo corregimos a float

# --------- App ---------
app = FastAPI(title="Iris Classification API")

# Carga modelo con ruta segura (relativa a este archivo)
MODEL_PATH = Path(__file__).parent / "iris_model.joblib"
model = joblib.load(MODEL_PATH)

species_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"  # <-- en lugar de "equisde" (si quieres dejalo, no afecta)
}

@app.get("/")
def root():
    return {"ok": True, "msg": "Iris API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_lenght,
        features.sepal_width,
        features.petal_lenght,
        features.petal_width
    ]])

    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0].max())

    return {
        "prediction": prediction,
        "species": species_map.get(prediction, "Unknown"),
        "confidence": confidence
    }