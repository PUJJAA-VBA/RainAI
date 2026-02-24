from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Extract values from frontend
    pressure = float(data["pressure"])
    maxtemp = float(data["maxtemp"])
    temparature = float(data["temparature"])
    mintemp = float(data["mintemp"])
    dewpoint = float(data["dewpoint"])
    humidity = float(data["humidity"])
    cloud = float(data["cloud"])
    sunshine = float(data["sunshine"])
    windspeed = float(data["windspeed"])

    features = np.array([[pressure, maxtemp, temparature,
                          mintemp, dewpoint, humidity,
                          cloud, sunshine, windspeed]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Rain Expected"
    else:
        result = "No Rain"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)