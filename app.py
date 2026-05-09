"""
Brain Tumor Detection - Web Application
========================================
A simple web interface to upload MRI images and classify them using a Deep Learning model.

Usage:
    python app.py
Then open your browser at http://localhost:5000
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# ==============================
# Configuration
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'brain_tumor_model.h5')
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ==============================
# Load Model
# ==============================
model = None

def get_model():
    """Load the trained model - loaded only once"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"[WARNING] Model not found at {MODEL_PATH}. Please train the model first.")
            return None
    return model

def preprocess_image(image_bytes):
    """
    Prepare the image for prediction
    - Resize to 224x224
    - Normalize values between 0 and 1
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor(image_bytes):
    """
    Predict the type of tumor from an MRI image
    
    Inputs: image bytes
    Outputs: dictionary containing the classification and confidence percentage
    """
    loaded_model = get_model()
    if loaded_model is None:
        return {"error": "Model not loaded. Please train the model first using the notebook."}
    
    img_array = preprocess_image(image_bytes)
    predictions = loaded_model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class]) * 100
    
    result = {
        "class": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 2),
        "all_predictions": {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }
    return result

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    """Home Page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction Endpoint - receives an image and returns the classification"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"}), 400
    
    try:
        image_bytes = file.read()
        result = predict_tumor(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Application Health Check"""
    model_loaded = get_model() is not None
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "classes": CLASS_NAMES
    })

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    print("=" * 60)
    print("  Brain Tumor Detection - Web Interface")
    print("=" * 60)
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Classes: {CLASS_NAMES}")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    # Try to load model at startup
    get_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
