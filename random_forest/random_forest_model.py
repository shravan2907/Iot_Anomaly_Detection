# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # For implementing the Random Forest model
from sklearn.preprocessing import LabelEncoder  # For encoding target labels with values between 0 and n_classes-1
from sklearn.metrics import confusion_matrix, classification_report  # For evaluating model performance
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For visualizing the confusion matrix

# Load the dataset
data_path = r'C:\Users\AKS\Desktop\Sem 7\PROJECT - 1\Iot_Anomaly_Detection\data\RT_IOT2022.csv'
df = pd.read_csv(data_path)  # Load the CSV dataset into a DataFrame

# Basic exploration of the dataset
print("Dataset Head:")
print(df.head())  # Display the first 5 rows of the dataset for an overview

print("\nDataset Info:")
print(df.info())  # Display information about the dataset, including data types and non-null counts

print("\nDataset Description:")
print(df.describe())  # Display basic statistical details like mean, std, min, and max for numerical columns

print("\nMissing Values in Dataset:")
print(df.isnull().sum())  # Display the count of missing values in each column

# Data Preprocessing
# Drop the 'Unnamed: 0' column if it's just an index
df = df.drop(columns=['Unnamed: 0'])  # Remove the unnecessary column that may have been added during dataset creation

# One-Hot Encoding for Categorical Variables
df = pd.get_dummies(df, columns=['proto', 'service'])  # Convert categorical variables 'proto' and 'service' into dummy/indicator variables

# Label encode the target variable
le = LabelEncoder()  # Initialize the label encoder
df['Attack_type'] = le.fit_transform(df['Attack_type'])  # Convert the 'Attack_type' target variable into numerical labels

# Split the dataset into features and target variable
X = df.drop(columns=['Attack_type'])  # Features (all columns except the target)
y = df['Attack_type']  # Target variable ('Attack_type')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% testing data

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest classifier with 100 trees
rf_model.fit(X_train, y_train)  # Train the model on the training data

# Make predictions
y_pred = rf_model.predict(X_test)  # Predict the target variable for the testing data

# Anomaly Detection Logic
anomalous_labels = [label for label in le.classes_ if label != 'Normal']  # Define which labels are considered anomalous
anomaly_predictions = [(index, pred) for index, pred in enumerate(y_pred) if le.inverse_transform([pred])[0] in anomalous_labels]  # Identify instances predicted as anomalies

# Count and print anomaly detections
if len(anomaly_predictions) > 0:
    print(f"\nAnomalies Detected: {len(anomaly_predictions)} instances.")
    detected_anomalies = [le.inverse_transform([pred])[0] for _, pred in anomaly_predictions]  # Get the actual labels of detected anomalies
    print(f"Types of Anomalies Detected: {set(detected_anomalies)}")  # Display the unique types of anomalies detected
    
    # Save the detected anomalies for future reference and analysis
    anomalies_df = pd.DataFrame(anomaly_predictions, columns=['Index', 'Predicted_Label'])  # Create a DataFrame for the detected anomalies
    anomalies_df['Predicted_Label'] = anomalies_df['Predicted_Label'].apply(lambda x: le.inverse_transform([x])[0])  # Convert numerical labels back to original labels
    anomalies_df.to_csv('detected_anomalies.csv', index=False)  # Save the anomalies to a CSV file
    print("\nDetected anomalies have been saved to 'detected_anomalies.csv'.")
else:
    print("\nNo anomalies detected.")  # Print if no anomalies were detected

# Evaluate the Model
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)  # Generate the confusion matrix to evaluate classification accuracy
print(conf_matrix)

print("\nClassification Report:")
class_report = classification_report(y_test, y_pred, target_names=le.classes_)  # Generate a classification report including precision, recall, and F1-score
print(class_report)

# Visualization
# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))  # Set the figure size for the plot
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')  # Plot the heatmap for the confusion matrix
plt.title('Confusion Matrix - Random Forest Model')  # Set the title for the heatmap
plt.xlabel('Predicted Labels')  # Set the x-axis label
plt.ylabel('True Labels')  # Set the y-axis label
plt.show()  # Display the heatmap

# Precision, Recall, and F1-Score Chart
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)  # Convert classification report to a dictionary
metrics_df = pd.DataFrame(report_dict).transpose()  # Convert the dictionary to a DataFrame for easier manipulation
classes = metrics_df.index[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg' from the metrics

x = np.arange(len(classes))  # Set up the x-axis for the plot
width = 0.25  # Width of the bars in the bar chart

plt.figure(figsize=(12, 6))  # Set the figure size for the plot
plt.bar(x - width, metrics_df.loc[classes, 'precision'], width, label='Precision')  # Plot precision scores for each class
plt.bar(x, metrics_df.loc[classes, 'recall'], width, label='Recall')  # Plot recall scores for each class
plt.bar(x + width, metrics_df.loc[classes, 'f1-score'], width, label='F1-Score')  # Plot F1-scores for each class

plt.xlabel('Classes')  # Set the x-axis label
plt.ylabel('Score')  # Set the y-axis label
plt.title('Precision, Recall, and F1-Score for Each Class - Random Forest Model')  # Set the title for the plot
plt.xticks(ticks=x, labels=classes, rotation=45)  # Set the tick labels for the x-axis
plt.legend()  # Add a legend to differentiate between precision, recall, and F1-score
plt.tight_layout()  # Adjust layout for better fit
plt.show()  # Display the bar chart
