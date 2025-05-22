import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv('IoT_Indoor_Air_Quality_Cleaned.csv')

# Label air quality based on thresholds
def label_air_quality(row):
    if (row['CO2 (ppm)'] > 1000 or
        row['PM2.5 (?g/m?)'] > 35 or
        row['PM10 (?g/m?)'] > 50 or
        row['TVOC (ppb)'] > 500 or
        row['CO (ppm)'] > 9):
        return 1  # Poor air quality
    else:
        return 0  # Good air quality

# Apply labeling function to the dataset
data['Air_Quality'] = data.apply(label_air_quality, axis=1)

# Define features and target
features = ['Temperature (?C)', 'Humidity (%)', 'CO2 (ppm)', 'PM2.5 (?g/m?)',
            'PM10 (?g/m?)', 'TVOC (ppb)', 'CO (ppm)']
target = 'Air_Quality'

X = data[features]
y = data[target]

# Scale features for better neural network training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Build the deep learning model using TensorFlow Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with optimizer, loss and metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, include verbose=1 for training output
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Make predictions on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Good', 'Poor']))

# Function to predict air quality on new samples
def check_air_quality(sample_input):
    """
    Predicts air quality ('Good' or 'Poor') based on sensor data.
    
    sample_input: List or array of feature values in order:
        ['Temperature (?C)', 'Humidity (%)', 'CO2 (ppm)', 'PM2.5 (?g/m?)',
         'PM10 (?g/m?)', 'TVOC (ppb)', 'CO (ppm)']
    """
    # Scale the input using the previously fitted scaler
    input_scaled = scaler.transform([sample_input])
    prediction_prob = model.predict(input_scaled)[0][0]
    result = "Poor" if prediction_prob >= 0.5 else "Good"
    print(f"Predicted Air Quality: {result} (Probability: {prediction_prob:.2f})")

# Example usage
sample_input = [23.0, 45.0, 800, 25, 40, 300, 5]
check_air_quality(sample_input)
