from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import os

app = Flask(__name__)  # Corrected __name_

# Load the model and class names at startup
model = load_model(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\plant_disease_model.h5')  # Added raw string for Windows path
with open(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\classes.json', 'r') as f:
    class_names = json.load(f)

# Function to preprocess the image
def preprocess_image(img, img_size=(128, 128)):
    img = cv2.resize(img, img_size)  # Resize image to match model input
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route to serve the HTML page
@app.route('/')
def home():
    svg_path = "{{url_for('daasvg.svg')}}"
    return render_template('index.html', svg=svg_path)  # Ensure 'index.html' is in the 'templates' folder

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "File upload failed"}), 400

    try:
        # Read and preprocess the image
        img_data = np.frombuffer(file.read(), np.uint8)  # Decode the file to raw numpy data
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)   # Convert to image format
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        processed_img = preprocess_image(img)

        # Make prediction
        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names.get(str(predicted_class_index), "Unknown")
        confidence = float(prediction[0][predicted_class_index])

        # Return prediction result
        return jsonify({
            "predicted_class": predicted_class_name,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':  # Corrected __name_
    app.run(debug=True)