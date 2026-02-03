import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset and preprocess
df = pd.read_csv('Rainfall.csv')
df.rename(str.strip, axis='columns', inplace=True)
df.replace({'yes': 1, 'no': 0}, inplace=True)
df.drop(['maxtemp', 'mintemp', 'day'], axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)

# Splitting features and target
features = df.drop(['rainfall'], axis=1)
target = df['rainfall']

# Standardizing features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train models
log_reg = LogisticRegression()
xgb = XGBClassifier()
svc = SVC(kernel='rbf', probability=True)

log_reg.fit(features_scaled, target)
xgb.fit(features_scaled, target)
svc.fit(features_scaled, target)

# Function to predict rainfall
def predict_rainfall(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    log_reg_pred = log_reg.predict(input_scaled)[0]
    xgb_pred = xgb.predict(input_scaled)[0]
    svc_pred = svc.predict(input_scaled)[0]
    
    return {
        'Logistic Regression': 'Rainfall' if log_reg_pred == 1 else 'No Rainfall',
        'XGBoost': 'Rainfall' if xgb_pred == 1 else 'No Rainfall',
        'SVC': 'Rainfall' if svc_pred == 1 else 'No Rainfall'
    }

# Streamlit UI
st.title("Rainfall Prediction App")

st.write("Enter the weather parameters to predict rainfall:")

# User input fields
input_values = []
feature_names = features.columns
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_values.append(value)

if st.button("Predict Rainfall"):
    predictions = predict_rainfall(input_values)
    st.write("### Predictions:")
    for model, result in predictions.items():
        st.write(f"{model}: {result}")
