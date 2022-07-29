from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle

app = Flask(__name__)
model = pickle.load(open("NB_classification.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']

        if type(news) is not int:

            data = re.sub('[^a-zA-Z0-9-\n]', ' ', news)
            data = re.sub('\s+', ' ', data)
            data = data.lower()

        output = model.predict([data])

        for i in output:
            f_output = i

    return render_template('home.html', prediction_text = "{}".format(f_output))

if __name__ == '__main__':
    app.run(debug=True)
