from sklearn.ensemble import IsolationForest
from data_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and preprocess the data
data_path = 'D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv'  # Adjust path as needed
X = load_and_preprocess_data(data_path)

# Split data into training and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Train Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train)

# Save the trained model
joblib.dump(model, 'isolation_forest_model.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Convert predictions: -1 (anomaly) to 1 and 1 (normal) to 0 for evaluation
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Create dummy labels (assuming all data is normal as placeholder)
y_test = [0] * len(y_pred)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# from sklearn.ensemble import IsolationForest
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from data_preprocessing import load_and_preprocess_data

# # Load and preprocess the data
# data_path = 'D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv'
# X, y = load_and_preprocess_data(data_path)
# if y is None:
#     print("No labels found in the dataset. Proceeding with unsupervised learning...")



# # Use the attack labels for evaluation
# attack_data = X[y != 'normal']
# normal_data = X[y == 'normal']
# y_attack = [1] * len(attack_data)  # Label attacks as anomalies (1)
# y_normal = [0] * len(normal_data)  # Label normal as normal (0)

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(normal_data, normal_data, test_size=0.3, random_state=42)

# # Balance the training data using SMOTE
# smote = SMOTE(sampling_strategy=1.0, random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Tune Isolation Forest
# model = IsolationForest(n_estimators=150, contamination=0.05, max_samples=0.8, random_state=42)
# model.fit(X_train_res)

# # Predict on test set
# y_pred = model.predict(X_test)
# y_pred = [1 if x == -1 else 0 for x in y_pred]

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
