import pandas as pd

# Load the dataset (using a smaller sample for inspection)
data = pd.read_csv('D:\Iot_Anomaly_Detection\data\RT_IOT2022.csv')

# View the first few rows and column names
print(data.head())
print(data.columns)