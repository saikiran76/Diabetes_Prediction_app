from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application

scaler = pickle.load(open("./model/standardScaler.pkl", "rb"))
model = pickle.load(open("./model/modelPred.pkl", "rb"))

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# route for Single data point prediction
@app.route("/predictions", methods=["GET", "POST"])
def predict():
    result = ""

    if request.method == "POST":
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BodyMI = -float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))
        
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BodyMI, DiabetesPedigreeFunction, Age]])
    
  
        predict = model.predict(new_data)
        if predict[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"
        return render_template("single_prediction.html", result=result)
    else:
        return render_template("home.html")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")



         
        