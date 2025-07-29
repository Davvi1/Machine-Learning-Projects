from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from utilities import DataPreprocessor, Winsorizer, TailClipper

app = FastAPI()

model = None  # define global variable for the model

class ModelInput(BaseModel):
    data: dict

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("voting_model_pipeline.pkl")
    print("✅ Model loaded")

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        df = pd.DataFrame([input_data.data])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]
        return {"prediction": int(pred), "probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
