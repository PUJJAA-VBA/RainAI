import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Rainfall.csv')
df.head()
df.rename(str.strip,
          axis='columns', 
          inplace=True)

df.columns
for col in df.columns:
  
  # Checking if the column contains
  # any null values
  if df[col].isnull().sum() > 0:
    val = df[col].mean()
    df[col] = df[col].fillna(val)
    
df.isnull().sum().sum()
df.groupby('rainfall').mean()
features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)
df.replace({'yes':1, 'no':0}, inplace=True)
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)
features = df.drop(['day', 'rainfall'], axis=1)
target = df.rainfall
X_train, X_val, \
    Y_train, Y_val = train_test_split(features,
                                      target,
                                      test_size=0.2,
                                      stratify=target,
                                      random_state=2)

# As the data was highly imbalancedwe will
# balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)
# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(3):
  models[i].fit(X, Y)

  print(f'{models[i]} : ')

  train_preds = models[i].predict_proba(X) 
  print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))

  val_preds = models[i].predict_proba(X_val) 
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
  print()
print(metrics.classification_report(Y_val,
                                    models[2].predict(X_val)))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

# Load dataset and preprocess
df = pd.read_csv('Rainfall.csv')
df.rename(str.strip, axis='columns', inplace=True)

df.replace({'yes': 1, 'no': 0}, inplace=True)
df.drop(['maxtemp', 'mintemp', 'day'], axis=1, inplace=True)

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Splitting features and target
features = df.drop(['rainfall'], axis=1)
target = df['rainfall']

expected_feature_count = features.shape[1]
print(f"Expected number of features: {expected_feature_count}")
# Load trained scaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Load trained models
log_reg = LogisticRegression()
xgb = XGBClassifier()
svc = SVC(kernel='rbf', probability=True)

log_reg.fit(features_scaled, target)
xgb.fit(features_scaled, target)
svc.fit(features_scaled, target)
def predict_rainfall(input_data):
    """
    Predicts whether rainfall will occur based on input features.
    """
    global expected_feature_count
    if len(input_data) != expected_feature_count:
        raise ValueError(f"Input data must have {expected_feature_count} features, but got {len(input_data)}")
    
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

# Example usage:
sample_input = [1012.8,18.4,18.0,72,49,9.3,80,26.3]  # Replace with actual feature values
predictions = predict_rainfall(sample_input)
print("Predicted Probabilities:", predictions)
