import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open(r"model.pkl", "rb"))

# Define the input fields for the Streamlit app
st.title("Weather Classification Prediction")

st.header("Input the following weather details:")

# Define input fields for numerical features
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")
precipitation = st.number_input("Precipitation (%)")
uv_index = st.number_input("UV Index")
visibility_km = st.number_input("Visibility (km)")
atmospheric_pressure = st.number_input("Atmospheric Pressure")

# Define input fields for categorical features
location = st.selectbox("Location", ["inland", "mountain", "coastal"])
cloud_cover = st.selectbox("Cloud Cover", ["clear", "partly cloudy", "overcast"])
season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'temperature': [temperature],
    'Humidity': [humidity],
    'Wind_Speed': [wind_speed],
    'Precipitation (%)': [precipitation],
    'UV Index': [uv_index],
    'Visibility_(km)': [visibility_km],
    'Atmospheric_Pressure': [atmospheric_pressure],
    'Location': [location],
    'Cloud Cover': [cloud_cover],
    'Season': [season]
})

# Preprocess and predict
if st.button("Predict Weather Type"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Weather Type: {prediction[0]}")

# To run the Streamlit app, execute the command:
# streamlit run app.py
