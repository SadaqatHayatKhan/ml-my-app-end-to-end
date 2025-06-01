from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Model loading
# Ensure this path is correct relative to where you run uvicorn
# If you run uvicorn from the 'backend' directory, this path is correct.
MODEL_PATH = "../ml/model.pkl" 

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at {MODEL_PATH}. The /predict endpoint will not work.")
    # You could raise an error here or handle it gracefully depending on requirements
    # raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
        else:
            # This case should ideally be handled by the startup check,
            # but as a safeguard within the function:
            return None 
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")

class TransactionFeatures(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                # Assuming 30 features as per the frontend
                "features": [0.0] * 30 
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict")
async def predict(transaction: TransactionFeatures):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or not found. Cannot make predictions.")
    
    if len(transaction.features) != 30: # Assuming your model expects 30 features
        raise HTTPException(status_code=400, detail=f"Invalid number of features. Expected 30, got {len(transaction.features)}")

    try:
        # Convert features to numpy array, suitable for scikit-learn models
        features_array = np.array(transaction.features).reshape(1, -1)
        
        # Make prediction
        # model.predict_proba returns probabilities for each class.
        # For binary classification, it's often [prob_class_0, prob_class_1]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Assuming class 1 is "fraud"
        fraud_probability = float(prediction_proba[1])
        
        # You can define your own threshold for classifying as fraud
        is_fraud = bool(fraud_probability > 0.5) 
        
        return {
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "input_features": transaction.features # Optional: echo back input
        }
    
    except AttributeError as e:
        # This might happen if the loaded model doesn't have predict_proba
        # or if the model object is not what's expected.
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}. Ensure the model has a 'predict_proba' method.")
    except Exception as e:
        # Catch any other unexpected errors during prediction
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

# To run the backend directly using `python main.py` (optional, for development)
if __name__ == "__main__":
    import uvicorn
    # Ensure uvicorn is installed: pip install uvicorn[standard]
    # Run from the 'backend' directory
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)