from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API (5 Features)")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model_5feat.pkl"

model = None

class Churn5FeatRequest(BaseModel):
    tenure: int
    Contract: str
    MonthlyCharges: float
    InternetService: str
    PaymentMethod: str


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        print("❌ Model file not found:", MODEL_PATH)
        return
    model = joblib.load(MODEL_PATH)
    print("✅ 5-feature model loaded")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(req: Churn5FeatRequest, threshold: float = 0.5):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python src/train_simple.py")

    df = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= threshold)

    return {"churn_probability": round(prob, 4), "churn_prediction": pred, "threshold": threshold}
