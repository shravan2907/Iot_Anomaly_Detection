# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Load the preprocessed dataset
# file_path = "D:\Iot_Anomaly_Detection\data\RT_IOT2022.csv"
# df = pd.read_csv(file_path)

# # Check the shape to confirm it has 81 features
# print(f"Shape of data: {df.shape}")  # Should be (rows, 81)

# # Select one row (e.g., the first row)
# sample_data = df.iloc[0].tolist()
# sample_data = sample_data.select_dtypes(include=['float64', 'int64'])

#     # Handle missing values, if any
# sample_data = sample_data.fillna(sample_data.mean())

# # Print the sample data
# print("Sample 81 feature values:", sample_data)
# print(len(sample_data))

import numpy as np

# Generate random input data with 81 features
test_input = (np.random.rand(81).tolist())

print(test_input)
