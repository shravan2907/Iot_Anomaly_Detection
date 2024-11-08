# src/save_model.py
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess_data

# Step 1: Load and preprocess the data
data_path = 'D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv'
X, y = load_and_preprocess_data(data_path)

# Step 2: Train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Step 3: Save the trained model using pickle
model_filename = 'isolation_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

# Step 4: Save the scaler used in preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler to the full dataset

scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved to {scaler_filename}")
