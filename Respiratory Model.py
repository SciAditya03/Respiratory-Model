!pip install pandas scikit-learn matplotlib seaborn tensorflow

import pandas as pd

# Load dataset (modify file path as needed)
data = pd.read_csv('Respiratory data sets.csv')
print(data.head())

def define_condition(row):
    if row['Temperature'] > 37.5:  # Fever threshold
        if row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
            return 'flu'  # High temperature + abnormal vitals
        return 'fever'  # High temperature, no severe abnormalities
    elif row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
        return 'cold'  # Abnormal vitals without high fever
    return 'healthy'  # Normal vitals and temperature

# Apply the function to each row
data['condition'] = data.apply(define_condition, axis=1)

# Check the distribution of the conditions
print(data['condition'].value_counts())

print(data.head())

# Define thresholds for fever and cold/flu (temperature in Celsius)
def define_labels(row):
    if row['Temperature'] > 37.5:  # 37.5°C is equivalent to 99.5°F
        if row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
            return 'flu'  # Flu-like symptoms with high temp and abnormal vitals
        return 'fever'  # High temperature, no severe vital irregularities
    elif row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
        return 'cold'  # Cold-like symptoms with abnormal vitals
    return 'healthy'  # Normal readings

# Apply the function to create output labels
data['condition'] = data.apply(define_labels, axis=1)

# Check the distribution of the labels
print(data['condition'].value_counts())

# Define thresholds for fever and cold/flu (temperature in Celsius)
def define_labels(row):
    if row['Temperature'] > 37.5:  # 37.5°C is equivalent to 99.5°F
        if row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
            return 'flu'  # Flu-like symptoms with high temp and abnormal vitals
        return 'fever'  # High temperature, no severe vital irregularities
    elif row['HR'] > 100 or row['RESP'] > 20 or row['SpO2'] < 95:
        return 'cold'  # Cold-like symptoms with abnormal vitals
    return 'healthy'  # Normal readings

# Apply the function to create output labels
data['condition'] = data.apply(define_labels, axis=1)

# Check the distribution of the labels
print(data['condition'].value_counts())

!pip install pandas scikit-learn matplotlib seaborn tensorflow


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Features: HR, PULSE, RESP, SpO2, Temperature
X = data[['HR', 'PULSE', 'RESP', 'SpO2', 'Temperature']]

# Target: condition (encoded as numbers for machine learning)
y = data['condition']

# Encode target labels as numeric
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'flu', 'fever', etc., to 0, 1, 2, etc.

# Save label mappings for later reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Define the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(len(label_mapping), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

import joblib

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_infection_model.pkl')

# Save the Neural Network
model.save('neural_network_infection_model.h5')

# Random Forest
rf_model = joblib.load('random_forest_infection_model.pkl')
new_data = [[90, 85, 18, 96, 37.0]]  # Replace with your input values
new_data_scaled = scaler.transform(new_data)
prediction = rf_model.predict(new_data_scaled)
print("Predicted Condition (Random Forest):", label_encoder.inverse_transform(prediction))

# Neural Network
from tensorflow.keras.models import load_model
model = load_model('neural_network_infection_model.h5')
nn_prediction = model.predict(new_data_scaled)
print("Predicted Condition (Neural Network):", label_encoder.inverse_transform([np.argmax(nn_prediction)]))

import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


import joblib

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_infection_model.pkl')
print("Random Forest model saved as 'random_forest_infection_model.pkl'")

# Save the Neural Network model
model.save('neural_network_infection_model.h5')
print("Neural Network model saved as 'neural_network_infection_model.h5'")

from google.colab import files
files.download('random_forest_infection_model.pkl')

files.download('neural_network_infection_model.h5')
