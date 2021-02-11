from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from pickle import load
from src.api.PredictionPipeline import PredictionPipeline
from tensorflow import keras

app = Flask(__name__, template_folder='src/frontend/templates')


class Pipeline:
    def __init__(self):
        with open("data/vocab.pl", "rb") as f:
            vocab = load(f)
        model = keras.models.load_model("./saved_models/")
        self.pp = PredictionPipeline(vocab, model)

    def predict(self, text):
        return 1 - self.pp.get_prediction(text)[0][0]


pipeline = None


@app.route('/')
def init():
    global pipeline
    pipeline = Pipeline()
    return redirect(url_for('get_text'))


@app.route('/home', methods=['POST', 'GET'])
def get_text():
    if request.method == 'POST':
        text = request.form['text']
        result = pipeline.predict(text)
        return render_template('home.html', text=text, result=result)
    else:
        return render_template('home.html', text="", result="")
