from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
print("🚀 Flask app is starting...")
@app.route('/')
def home():
    
    return render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    # Get all 15 input features from the form
    features = [
        float(request.form['male']),
        float(request.form['age']),
        float(request.form['education']),
        float(request.form['currentSmoker']),
        float(request.form['cigsPerDay']),
        float(request.form['BPMeds']),
        float(request.form['prevalentStroke']),
        float(request.form['prevalentHyp']),
        float(request.form['diabetes']),
        float(request.form['totChol']),
        float(request.form['sysBP']),
        float(request.form['diaBP']),
        float(request.form['BMI']),
        float(request.form['heartRate']),
        float(request.form['glucose'])
    ]

    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'10-Year CHD Risk Prediction: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
    
