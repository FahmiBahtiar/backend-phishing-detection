"""
Script untuk konversi model scikit-learn ke ONNX
"""
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# Path model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "rf_model_25_features.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model", "rf_model_25_features.onnx")

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Model type: {type(model)}")
print(f"Model classes: {model.classes_}")

# Define input type (25 features)
initial_type = [('float_input', FloatTensorType([None, 25]))]

# Convert to ONNX dengan opsi untuk menyertakan probability
print("Converting to ONNX...")
onnx_model = convert_sklearn(
    model, 
    initial_types=initial_type,
    options={id(model): {'zipmap': False}}  # Disable zipmap for easier probability access
)

# Save ONNX model
print(f"Saving ONNX model to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Verify the conversion
print("\n--- Verification ---")
import onnxruntime as ort

# Load ONNX model
sess = ort.InferenceSession(OUTPUT_PATH)

# Test input
test_input = np.array([[-1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1]], dtype=np.float32)

# Get input name
input_name = sess.get_inputs()[0].name
print(f"Input name: {input_name}")

# Get output names
output_names = [o.name for o in sess.get_outputs()]
print(f"Output names: {output_names}")

# Run prediction
results = sess.run(None, {input_name: test_input})
print(f"Prediction (label): {results[0]}")
print(f"Probabilities: {results[1]}")

# Compare with original model
original_pred = model.predict(test_input)
original_proba = model.predict_proba(test_input)
print(f"\nOriginal model prediction: {original_pred}")
print(f"Original model probabilities: {original_proba}")

# Check file size
onnx_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
pkl_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\nOriginal .pkl size: {pkl_size:.2f} MB")
print(f"ONNX model size: {onnx_size:.2f} MB")

print("\n✅ Conversion successful!")
