from flask import Flask, render_template, request
from flask_cors import CORS
from pickle import load
from api.Preprocessing import Preprocessing
from api.Postprocessing import Postprocessing
from api.Embedding import Embedding
from api.PredictionPipeline import PredictionPipeline
from tensorflow import keras

app = Flask(__name__)


## init
prep = Preprocessing()
post = Postprocessing()
emb = Embedding()
with open("../../data/vocab.pl", "rb") as f:
    vocab = load(f)
model = keras.models.load_model("../../model2.h5")
pp = PredictionPipeline(vocab, model)
print("Init done.")

@app.route('/', methods=['POST', 'GET'])
def get_text():
    if request.method == 'POST':
        text = request.form['text']
        result = 1. - pp.get_prediction(text)[0][0]
        return render_template('home.html', text=text, result=result)
    else:
        return render_template('home.html', text="", result="")