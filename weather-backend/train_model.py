import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# Load CSV
data = pd.read_csv("Rainfall.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Convert rainfall yes/no to 1/0
data['rainfall'] = data['rainfall'].map({'yes': 1, 'no': 0})

# Define features
X = data[['pressure',
          'maxtemp',
          'temparature',
          'mintemp',
          'dewpoint',
          'humidity',
          'cloud',
          'sunshine',
          'windspeed']]

# Target
y = data['rainfall']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use classifier (since rainfall is yes/no)
model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")