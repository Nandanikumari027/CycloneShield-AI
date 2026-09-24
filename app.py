from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load trained model
model = joblib.load("model/cyclone_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    wind_speed = float(request.form["wind_speed"])
    rainfall = float(request.form["rainfall"])
    distance_from_coast = float(request.form["distance_from_coast"])
    infrastructure_age = float(request.form["infrastructure_age"])
    population_density = float(request.form["population_density"])
    cyclone_intensity = float(request.form["cyclone_intensity"])

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "wind_speed": wind_speed,
        "rainfall": rainfall,
        "distance_from_coast": distance_from_coast,
        "infrastructure_age": infrastructure_age,
        "population_density": population_density,
        "cyclone_intensity": cyclone_intensity
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Model confidence
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(max(probabilities) * 100, 2)

    # Risk level
    if prediction == 0:
        risk = "LOW RISK"
    elif prediction == 1:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    # Explanation
    reasons = []

    if wind_speed >= 140:
        reasons.append("Very high wind speed")
    elif wind_speed >= 100:
        reasons.append("High wind speed")

    if rainfall >= 200:
        reasons.append("Heavy rainfall")
    elif rainfall >= 120:
        reasons.append("Moderate to heavy rainfall")

    if distance_from_coast <= 30:
        reasons.append("Very close to coastline")
    elif distance_from_coast <= 70:
        reasons.append("Relatively close to coastline")

    if infrastructure_age >= 40:
        reasons.append("Older infrastructure")

    if population_density >= 1500:
        reasons.append("High population density")

    if cyclone_intensity >= 5:
        reasons.append("Very strong cyclone intensity")
    elif cyclone_intensity >= 4:
        reasons.append("Strong cyclone intensity")
    # Recommended actions
    if risk == "HIGH RISK":
        recommendations = [
            "Evacuate highly vulnerable areas if advised by authorities",
            "Inspect critical infrastructure",
            "Prepare emergency response teams"
            ]

    elif risk == "MEDIUM RISK":
        recommendations = [
            "Monitor weather conditions closely",
            "Inspect vulnerable infrastructure",
            "Keep emergency resources ready"
            ]

    else:
        recommendations = [
            "Continue monitoring weather conditions",
            "Maintain normal preparedness",
            "Review local disaster-response plans"
            ]
    return render_template(
        "index.html",
        prediction=risk,
        confidence=confidence,
        reasons=reasons,
        wind_speed=wind_speed,
        rainfall=rainfall,
        distance_from_coast=distance_from_coast,
        infrastructure_age=infrastructure_age,
        population_density=population_density,
        cyclone_intensity=cyclone_intensity,
        recommendations=recommendations
    )


# Weather API function
def get_weather():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 25.2138,
        "longitude": 75.8648,
        "current": "wind_speed_10m,precipitation",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        "wind_speed": data["current"]["wind_speed_10m"],
        "rainfall": data["current"]["precipitation"]
    }
@app.route("/weather")
def weather():
    try:
        weather_data = get_weather()
        return weather_data
    except Exception as e:
        return {
            "error": str(e)
        }, 500


if __name__ == "__main__":
    app.run(debug=True)