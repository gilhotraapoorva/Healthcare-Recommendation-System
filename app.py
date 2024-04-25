import numpy as np
from flask import Flask, request, jsonify, render_template

#Create flask app
app = Flask(__name__)

# Load the h5 model 
model = h5.load(open("model.h5", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",  method = ["POST"])
def predict():
    
    prediction = model.predict(parameters)
    return render_template("index.html",prediction_text = "The disease you are suffering from is {}".format{prediction})


if __name__ == '__main__':
    app.run(debug = True)