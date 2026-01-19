"""
FastAPI Backend for Phishing Detection
Serverless deployment for Vercel
"""

import os
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Use joblib for better model loading compatibility
try:
    import joblib
except ImportError:
    import pickle as joblib

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="API untuk mendeteksi URL phishing menggunakan Random Forest model",
    version="1.0.0"
)

# CORS middleware untuk akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model saat startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rf_model_25_features.pkl")

model = None

def load_model():
    """Load model dari file .pkl using joblib"""
    global model
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return model


# 25 fitur yang digunakan model (urutan sesuai training)
FEATURE_NAMES = [
    "SSLfinal_State",
    "URL_of_Anchor",
    "Prefix_Suffix",
    "web_traffic",
    "having_Sub_Domain",
    "Request_URL",
    "Links_in_tags",
    "Domain_registeration_length",
    "SFH",
    "Google_Index",
    "age_of_domain",
    "Page_Rank",
    "having_IPhaving_IP_Address",
    "Statistical_report",
    "DNSRecord",
    "Shortining_Service",
    "Abnormal_URL",
    "URLURL_Length",
    "having_At_Symbol",
    "on_mouseover",
    "HTTPS_token",
    "double_slash_redirecting",
    "port",
    "Links_pointing_to_page",
    "Redirect"
]


class PredictionInput(BaseModel):
    """
    Input schema untuk prediksi phishing.
    Menerima array 25 fitur numerik.
    Nilai fitur: -1 (phishing indicator), 0 (suspicious), 1 (legitimate indicator)
    """
    features: List[float] = Field(
        ...,
        min_length=25,
        max_length=25,
        description="Array of 25 numerical features extracted from URL"
    )


class PredictionOutput(BaseModel):
    """Output schema untuk hasil prediksi"""
    prediction: int = Field(..., description="Prediction result: -1 (phishing) or 1 (legitimate)")
    label: str = Field(..., description="Human-readable label: 'phishing' or 'legitimate'")
    probability: float = Field(..., description="Confidence score (0-1)")
    message: str = Field(..., description="Description of the result")


class HealthResponse(BaseModel):
    """Response untuk health check"""
    status: str
    message: str
    model_loaded: bool
    features_count: int


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Memeriksa status API dan model.
    """
    try:
        loaded_model = load_model()
        return HealthResponse(
            status="healthy",
            message="Phishing Detection API is running",
            model_loaded=loaded_model is not None,
            features_count=len(FEATURE_NAMES)
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=str(e),
            model_loaded=False,
            features_count=len(FEATURE_NAMES)
        )


@app.get("/features")
async def get_feature_names():
    """
    Mendapatkan daftar nama fitur yang diharapkan model.
    """
    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES),
        "description": "Nilai fitur: -1 (phishing indicator), 0 (suspicious), 1 (legitimate indicator)"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_phishing(input_data: PredictionInput):
    """
    Melakukan prediksi phishing berdasarkan fitur URL.
    
    - **features**: Array 25 nilai numerik dari ekstraksi fitur URL
    
    Response time target: < 250ms
    """
    try:
        # Load model
        loaded_model = load_model()
        
        # Prepare input
        features_array = np.array(input_data.features).reshape(1, -1)
        
        # Predict
        prediction = loaded_model.predict(features_array)[0]
        
        # Get probability
        probability_scores = loaded_model.predict_proba(features_array)[0]
        
        # Model uses -1 for phishing and 1 for legitimate
        if prediction == -1:
            label = "phishing"
            # Probability for phishing class
            probability = float(probability_scores[0])
            message = "⚠️ URL ini terdeteksi sebagai PHISHING. Hindari mengakses URL ini."
        else:
            label = "legitimate"
            # Probability for legitimate class
            probability = float(probability_scores[1])
            message = "✅ URL ini terdeteksi sebagai LEGITIMATE (aman)."
        
        return PredictionOutput(
            prediction=int(prediction),
            label=label,
            probability=probability,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
