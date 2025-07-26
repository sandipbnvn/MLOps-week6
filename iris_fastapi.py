from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import sys
import os

app = FastAPI(title="Iris Classifier API")
sys.path.append(os.path.join(os.path.dirname(__file__), 'artifacts'))

# Load model
model_data = joblib.load("artifacts/model.joblib")
model = model_data['model']

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!v3"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"predicted_class": prediction}
