#streamlit env
import streamlit as st
import numpy as np
import pandas as pd
import joblib


@st.cache_data
def load_model():
    model_path = r"NPKModel.pkl"  
    # made by saanvi 
    model = joblib.load(model_path)
    return model

loaded_model = load_model()

st.title('Crop Recommendation Predictor')

# Input form for the user to enter the environmental conditions
with st.form("prediction_form"):
    st.subheader("Enter the conditions for crop recommendation:")
    #made by saanvi
    N = st.number_input("Nitrogen", min_value=0.0, value=90.0, step=1.0)
    P = st.number_input("Phosphorus", min_value=0.0, value=42.0, step=1.0)
    K = st.number_input("Potassium", min_value=0.0, value=43.0, step=1.0)
    temperature = st.number_input("Temperature", min_value=-10.0, value=20.9, step=0.1)
    humidity = st.number_input("Humidity", min_value=0.0, value=75.0, step=1.0)
    pH = st.number_input("pH", min_value=0.0, value=5.5, step=0.1)
    rainfall = st.number_input("Rainfall", min_value=0.0, value=220.0, step=1.0)
    
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    predictions = loaded_model.predict(input_data)
    st.subheader(f"Predictions: {predictions}")
