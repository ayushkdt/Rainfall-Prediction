from flask import Flask, request, render_template 
import pickle
import numpy as np 

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    day = float(request.form['day'])
    pressure = float(request.form['pressure'])
    maxtemp = float(request.form['maxtemp'])
    temperature = float(request.form['temperature'])
    mintemp = float(request.form['mintemp'])
    dewpoint = float(request.form['dewpoint'])
    humidity = float(request.form['humidity'])
    cloud = float(request.form['cloud'])
    rainfall = float(request.form['rainfall'])
    sunshine = float(request.form['sunshine'])
    winddirection = float(request.form['winddirection'])
    windspeed = float(request.form['windspeed'])

    # Create a feature array for prediction
    features = np.array([[day, pressure, maxtemp, temperature, mintemp, dewpoint, humidity, cloud, rainfall, sunshine, winddirection, windspeed]])

    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Assuming binary classification: 0 = No Rain, 1 = Rain
    if prediction[0] == 1:
        rain_probability = prediction_proba[0][1] * 100  # Probability of rain
        prediction_text = f"It will rain today with a {rain_probability:.2f}% chance."
    else:
        rain_probability = prediction_proba[0][0] * 100  # Probability of no rain
        prediction_text = f"It won't rain today with a {rain_probability:.2f}% chance."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
