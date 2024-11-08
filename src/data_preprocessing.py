import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop unnecessary columns (e.g., 'Unnamed: 0', 'Attack_type')
    data = data.drop(columns=['Unnamed: 0', 'Attack_type'], errors='ignore')  # 'errors' to handle missing columns

    # Select only numeric columns
    data = data.select_dtypes(include=['float64', 'int64'])

    # Handle missing values, if any
    data = data.fillna(data.mean())

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)
    
    # Save the scaler to file
    joblib.dump(scaler, 'scaler.pkl')
    
    return features_scaled
# def load_and_preprocess_data(file_path):
#     # Load the dataset
#     data = pd.read_csv(file_path)

#     # Drop unnecessary columns
#     if 'Unnamed: 0' in data.columns:
#         data = data.drop(columns=['Unnamed: 0'])
    
#     # Separate features and labels
#     if 'Attack_type' in data.columns:
#         y = data['Attack_type']
#         data = data.drop(columns=['Attack_type'])
#     else:
#         y = None

#         # Separate numeric and non-numeric columns
#     numeric_columns = X.select_dtypes(include=['float64', 'int64'])
#     non_numeric_columns = X.select_dtypes(exclude=['float64', 'int64'])

#     # Fill missing values for numeric columns
#     numeric_columns = numeric_columns.fillna(numeric_columns.mean())

#     # Combine numeric and non-numeric columns
#     X = pd.concat([numeric_columns, non_numeric_columns], axis=1)

#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
    
#     return features_scaled, y  # Now returns both features and labels


