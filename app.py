from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Load the trained model (NPKModel.pkl) during startup
model_path = r'C:\saanvi_code\Agri-link-main\sources\NPKModel.pkl'

try:
    # Load the model using joblib
    loaded_model = joblib.load(model_path)
    
    # Check if the loaded model has a predict method
    if hasattr(loaded_model, 'predict'):
        logging.debug(f"Model loaded successfully. Type: {type(loaded_model)}")
    else:
        raise ValueError("Loaded model does not have a 'predict' method.")
        
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    loaded_model = None  # Ensure loaded_model is None if loading fails

# Video dictionary for rendering in HTML templates
video = {
    "url": "/static/video/sup.mp4"
}

@app.route('/')
def home():
    return render_template('main.html', vid=video)

@app.route('/farm')
def farm():
    return render_template('farm.html', vid=video)

@app.route('/weather')
def weather():
    return render_template('Weatherapp.html', vid=video)

@app.route('/price')
def price():
    return render_template('PriceObserver.html', vid=video)

@app.route('/soil')
def soil():
    return render_template('soilt.html', vid=video)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': 'Model is not loaded properly.'}), 500

    try:
        # Parse incoming JSON data
        data = request.json
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        pH = float(data.get('pH'))
        rainfall = float(data.get('rainfall'))

        # Prepare the data for the model
        input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Perform prediction
        prediction = loaded_model.predict(input_data)  # Use predict for classification

        # Return prediction result as JSON
        return jsonify({'prediction': prediction.tolist()})  # Convert prediction to list

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
