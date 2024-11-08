# Test the data preprocessing
from data_preprocessing import load_and_preprocess_data

data_path = 'D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv'
X, y = load_and_preprocess_data(data_path)
print(X.head())  # Print first 5 rows of the processed features
print(y[:5])     # Print first 5 labels (if any)
