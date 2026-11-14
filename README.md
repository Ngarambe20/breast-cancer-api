# Breast Cancer Prediction API

A Machine Learning API built with FastAPI and scikit-learn that predicts whether a breast tumor is malignant or benign based on 30 numerical features.

## Tech Stack
- Python
- FastAPI / Uvicorn
- Scikit-Learn
- Joblib

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`
6. Open the interactive docs at `http://127.0.0.1:8000/docs`

## API Endpoints
- `GET /health`: Checks if the API is running.
- `POST /predict`: Accepts a JSON payload with a list of 30 floats and returns the prediction.