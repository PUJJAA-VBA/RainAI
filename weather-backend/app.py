from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests

API_KEY = "7cb98fede82fd6e2b308e6ed97a7f887"

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

@app.route('/get-weather/<city>')
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
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

    return weather_data



if __name__ == "__main__":
    app.run(debug=True)