from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data = np.array(data).reshape(1, -1)

    # Scale the data
    data_scaled = scaler.transform(data)

    # Get prediction from Isolation Forest
    pred = model.predict(data_scaled)

    # Convert prediction: -1 (anomaly) to 1 and 1 (normal) to 0
    result = 1 if pred[0] == -1 else 0

    return jsonify({'anomaly': result})

if __name__ == '__main__':
    app.run(debug=True)
# # from flask import Flask, request, jsonify
# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import IsolationForest
# # import joblib

# # # Load the scaler and model
# # scaler = joblib.load('scaler.pkl')
# # model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# # app = Flask(__name__)

# # # Load dataset to simulate an attack sample
# # data = pd.read_csv('D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv')
# # # Add this code right after loading the dataset
# # from sklearn.preprocessing import LabelEncoder

# # # Encode the 'proto' column (and others if needed)
# # label_encoder = LabelEncoder()
# # data['proto'] = label_encoder.fit_transform(data['proto'])

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Simulate a known attack (e.g., DOS_SYN_Hping)
# #     attack_data_sample = data[data['Attack_type'] == 'DOS_SYN_Hping'].drop(['Attack_type'], axis=1).iloc[0].values
    
# #     # Use this attack sample for prediction
# #     data_scaled = scaler.transform([attack_data_sample])
    
# #     # Get prediction from Isolation Forest
# #     pred = model.predict(data_scaled)

# #     # Convert prediction: -1 (anomaly) to 1 and 1 (normal) to 0
# #     result = 1 if pred[0] == -1 else 0

# #     return jsonify({'anomaly': result})

# # if __name__ == '__main__':
# #     app.run(debug=True)
# import joblib
# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify

# # Load the pre-trained model and scaler
# model = joblib.load('isolation_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Load dataset to extract feature columns
# data_path = 'D:/Iot_Anomaly_Detection/data/RT_IOT2022.csv'
# data = pd.read_csv(data_path)

# # Drop unnecessary columns
# data = data.drop(columns=['Unnamed: 0', 'Attack_type'], errors='ignore')

# # Filter numeric columns for scaling
# numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the input JSON
#         attack_data_sample = request.json.get('input_data')
        
#         # Create DataFrame from the input
#         attack_data_sample = pd.DataFrame([attack_data_sample])

#         # Ensure we only use numeric columns
#         attack_data_sample_cleaned = attack_data_sample[numeric_columns].fillna(data.mean())

#         # Check if there's any valid numeric data left
#         if attack_data_sample_cleaned.empty:
#             raise ValueError("No numeric data available after cleaning input.")

#         # Scale the numeric data
#         data_scaled = scaler.transform(attack_data_sample_cleaned)

#         # Predict using the loaded model
#         prediction = model.predict(data_scaled)

#         # Map the prediction (-1: Anomaly, 1: Normal)
#         result = 'Anomaly' if prediction[0] == -1 else 'Normal'
        
#         return jsonify({'prediction': result})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
