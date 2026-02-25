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
    print("Model loaded successfully")
except:
    print("Model loading failed. Check model.pkl file.")
    model = None


# ==========================
# Helper Function (Weather Fetch)
# ==========================
def fetch_weather(city):

    if API_KEY is None:
        return None, "API key missing"

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=3)

        if response.status_code != 200:
            return None, "City not found or API error"

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

        return weather_data, None

    except Exception as e:
        return None, str(e)


# ==========================
# Live Weather API
# ==========================
@app.route("/live-weather/<city>")
def live_weather(city):

    weather_data, error = fetch_weather(city)

    if error:
        return jsonify({"error": error}), 400

    return jsonify(weather_data)


# ==========================
# Prediction API (Using Live Weather)
# ==========================
@app.route("/predict/<city>")
def predict(city):

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    weather_data, error = fetch_weather(city)

    if error:
        return jsonify({"error": error}), 400

    try:
        features = np.array([[
            weather_data["pressure"],
            weather_data["maxtemp"],
            weather_data["temparature"],
            weather_data["mintemp"],
            0,  # dewpoint placeholder
            weather_data["humidity"],
            weather_data["cloud"],
            0,  # sunshine placeholder
            weather_data["windspeed"]
        ]])

        # prediction = model.predict(features)[0]
        prediction = int(model.predict(features)[0])

        result_map = {
            1: "Rain Expected",
            0: "No Rain"
        }

        return jsonify({
            "prediction": result_map.get(prediction, "Unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    app.run(debug=True, threaded=True)