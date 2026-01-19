"""
FastAPI Backend for Phishing Detection
Serverless deployment for Vercel - Using ONNX Runtime
"""

import os
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="API untuk mendeteksi URL phishing menggunakan Random Forest model (ONNX)",
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

# Load model saat startup - using ONNX
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rf_model_25_features.onnx")

session = None

def load_model():
    """Load ONNX model"""
    global session
    if session is None:
        try:
            session = ort.InferenceSession(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return session


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
        loaded_session = load_model()
        return HealthResponse(
            status="healthy",
            message="Phishing Detection API is running (ONNX)",
            model_loaded=loaded_session is not None,
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
        # Load ONNX model
        loaded_session = load_model()
        
        # Prepare input (ONNX expects float32)
        features_array = np.array(input_data.features, dtype=np.float32).reshape(1, -1)
        
        # Get input name from model
        input_name = loaded_session.get_inputs()[0].name
        
        # Run prediction
        results = loaded_session.run(None, {input_name: features_array})
        
        # results[0] = label, results[1] = probabilities
        prediction = int(results[0][0])
        probabilities = results[1][0]  # Array of probabilities for each class
        
        # Model uses -1 for phishing (index 0) and 1 for legitimate (index 1)
        if prediction == -1:
            label = "phishing"
            probability = float(probabilities[0])  # Probability for phishing class
            message = "⚠️ URL ini terdeteksi sebagai PHISHING. Hindari mengakses URL ini."
        else:
            label = "legitimate"
            probability = float(probabilities[1])  # Probability for legitimate class
            message = "✅ URL ini terdeteksi sebagai LEGITIMATE (aman)."
        
        return PredictionOutput(
            prediction=prediction,
            label=label,
            probability=round(probability, 4),
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
