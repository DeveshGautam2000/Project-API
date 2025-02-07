from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel

with open("xgboost_model_top5.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Input Schema with correct feature names
class InputData(BaseModel):
    Age: int
    Annual_Income: float  
    Health_Score: float
    Credit_Score: float
    Insurance_Duration: float

@app.post("/predict/")
async def predict(data: InputData, request: Request):
    try:
        # Log received data
        print("Received data:", await request.json())

        # Convert input to DataFrame
        df_input = pd.DataFrame([data.dict()])

        # Ensure column names match exactly with training data
        expected_columns = ["Age", "Annual_Income", "Health_Score", "Credit_Score", "Insurance_Duration"]
        df_input = df_input[expected_columns]

        # Scale numerical features
        df_input[expected_columns] = scaler.transform(df_input)

        # Make prediction
        prediction = model.predict(df_input)
        return {"predicted_premium": float(prediction[0])}

    except Exception as e:
        print("Prediction Error:", str(e))  # Log error details
        raise HTTPException(status_code=400, detail=str(e))