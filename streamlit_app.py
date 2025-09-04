import streamlit as st
import numpy as np
import joblib

# Load the trained model
MODEL_FILE = "model.joblib"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found.")
        return None

# Load the model
model = load_model()

# Title of the app
st.title("AdaBoost Classifier Prediction App")

# Define 18 input fields for the user
st.header("Enter Feature Values")
features = []

for i in range(18):  
    feature_value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(feature_value)

# Convert user input into a NumPy array
input_data = np.array(features).reshape(1, -1)

# Predict button
if st.button("Predict"):
    if model:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.error("Model not loaded properly. Check file path or re-train the model.")

