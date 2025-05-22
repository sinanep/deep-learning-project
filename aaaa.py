import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Air Quality Checker", layout="centered")

st.title("🌫️ Air Quality Checker (Deep Learning)")
st.markdown("This app uses a trained deep learning model to predict whether the indoor air quality is **Good** or **Poor** based on sensor data.")

# Load and preprocess dataset
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('IoT_Indoor_Air_Quality_Cleaned.csv')

    # Label air quality
    def label_air_quality(row):
        if (row['CO2 (ppm)'] > 1000 or
            row['PM2.5 (?g/m?)'] > 35 or
            row['PM10 (?g/m?)'] > 50 or
            row['TVOC (ppb)'] > 500 or
            row['CO (ppm)'] > 9):
            return 1  # Poor
        else:
            return 0  # Good

    data['Air_Quality'] = data.apply(label_air_quality, axis=1)

    features = ['Temperature (?C)', 'Humidity (%)', 'CO2 (ppm)', 'PM2.5 (?g/m?)',
                'PM10 (?g/m?)', 'TVOC (ppb)', 'CO (ppm)']
    target = 'Air_Quality'

    X = data[features]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

    return model, scaler

model, scaler = load_and_train_model()

# Input form
with st.form("input_form"):
    st.subheader("📥 Enter Sensor Readings")
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=23.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=45.0)
    co2 = st.number_input("CO₂ (ppm)", min_value=0.0, max_value=5000.0, value=800.0)
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=25.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=40.0)
    tvoc = st.number_input("TVOC (ppb)", min_value=0.0, max_value=1000.0, value=300.0)
    co = st.number_input("CO (ppm)", min_value=0.0, max_value=100.0, value=5.0)

    submitted = st.form_submit_button("Check Air Quality")

if submitted:
    input_data = np.array([[temp, humidity, co2, pm25, pm10, tvoc, co]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]
    result = "Poor" if prob >= 0.5 else "Good"
    
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Predicted Air Quality:** `{result}`")
    st.markdown(f"**Model Confidence:** `{prob:.2f}`")

    st.success("Thank you for using the Air Quality Checker!")
