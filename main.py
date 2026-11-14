from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize app
app = FastAPI(title="Breast Cancer Prediction API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Load model
model = joblib.load('model.joblib')

# 3. Define input schema
class PatientData(BaseModel):
    features: list[float]

# 4. Root endpoint (just so the base URL doesn't show "Not Found")
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# 5. Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# 6. Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    # Validate that the user sent exactly 30 features
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Input must contain exactly 30 features.")
        
    # Convert list to numpy array and reshape for the model
    input_array = np.array(data.features).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    
    # Extract confidence
    confidence = float(probabilities[prediction])
    
    # Map result
    result = "Malignant" if prediction == 0 else "Benign"
    
    return {
        "prediction": result,
        "prediction_code": int(prediction),
        "confidence": round(confidence, 4)
    }