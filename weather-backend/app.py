from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")

app = Flask(__name__)
CORS(app)

# Load ML model safely
try:
    model = joblib.load("model.pkl")
except:
    print("Model loading failed. Check model.pkl file.")
    model = None


# ==========================
# Prediction API
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json

        pressure = float(data["pressure"])
        maxtemp = float(data["maxtemp"])
        temparature = float(data["temparature"])
        mintemp = float(data["mintemp"])
        dewpoint = float(data["dewpoint"])
        humidity = float(data["humidity"])
        cloud = float(data["cloud"])
        sunshine = float(data["sunshine"])
        windspeed = float(data["windspeed"])

        features = np.array([[
            pressure,
            maxtemp,
            temparature,
            mintemp,
            dewpoint,
            humidity,
            cloud,
            sunshine,
            windspeed
        ]])

        prediction = model.predict(features)[0]

        result_map = {
            1: "Rain Expected",
            0: "No Rain"
        }

        return jsonify({
            "prediction": result_map.get(prediction, "Unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================
# Live Weather API Integration
# ==========================
@app.route("/get-weather/<city>")
def get_weather(city):

    if API_KEY is None:
        return jsonify({"error": "API key missing"}), 500

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=2)

        if response.status_code != 200:
            return jsonify({"error": "City not found or API error"}), 404

        data = response.json()

        weather_data = {
            "pressure": data["main"]["pressure"],
            "maxtemp": data["main"]["temp_max"],
            "temparature": data["main"]["temp"],
            "mintemp": data["main"]["temp_min"],
            "humidity": data["main"]["humidity"],
            "cloud": data["clouds"]["all"],
            "windspeed": data["wind"]["speed"]
        }

        return jsonify(weather_data)

    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather API timeout"}), 504

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    app.run(debug=True)