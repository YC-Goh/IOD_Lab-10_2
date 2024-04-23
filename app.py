from flask import Flask, request, jsonify, render_template
import pandas as pd

import joblib
from utils import preprocessor

app = Flask(__name__)
model = joblib.load('./model.joblib')

##  Your basic landing page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input = request.form['userinput']
    predicted_sentiment = model.predict(pd.Series([input]))[0]
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('index.html', prediction=f'Predicted sentiment of "{input}" is {output}.')

##  name == main is executed only when python executes app.py
##  This won't get executed when Flask is called to compile your website instead
if __name__ == "__main__":
    ##  Since we don't call Flask to compile the website during debug mode
    ##  This stands in for that call
    ##  By the way -- note that Flask + GUnicorn seems to compile webpages dynamically
    ##  As in, only when a certain route is requested, then Flask will compile the webpage using the template and any input data
    ##  Before passing onto GUnicorn to serve to users
    ##  However Flask uses templates instead of html components so the server-side compilation can be much faster if the application itself is not too dynamic
    app.run(debug=True)