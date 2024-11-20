from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import os

app = Flask(__name__)

model = load_model(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\plant_disease_model.h5')
with open(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\classes.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
with open(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\cure.json', 'r', encoding='utf-8') as f:
    cure_info = json.load(f)

def preprocess_image(img, img_size=(128, 128)):
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

predicted_class_var = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predicted_class_var

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "File upload failed"}), 400

    try:
        img_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names.get(str(predicted_class_index), "Unknown")
        confidence = float(prediction[0][predicted_class_index])

        predicted_class_var = predicted_class_name

        cure = cure_info.get(predicted_class_name, "No cure information available")


        print(f"Predicted Class: {predicted_class_name}, Cure: {cure}")

        return jsonify({
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "cure": cure
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("Keys in cure_info:", list(cure_info.keys()))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
