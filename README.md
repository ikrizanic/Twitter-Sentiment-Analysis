# Twitter Sentiment Analysis

The core of the project is NLP analysis of Twitter posts. LSTM based model is trained on Kaggle dataset and is used 
for prediction of Tweet sentiment. Simple web app uses complete pipeline which 
gives sentiment evaluation based on given text.
<br>
<br>
The idea of project is to compare results of different models, but for now only one model is implemented.

# Web app

#### A working version of basic flask app is online and available with [this link](https://twitter-sentiment-analysis-app.herokuapp.com/).
App is currently deployed on _Heroku_, so it can be slow to load at the moments
## Future work
- implement more models for comparison of predicted sentiments
- implement React.js frontend for better looks

## About implementation

Currently, only one basic LSTM model is implemented, but more implementations will follow.
<br>All parts of code related to the model and NLP are located in the _api_ folder. The rest of code is used
for basic _Flask_ app and _Bootstrap_ frontend.
### Prediction pipeline
Returns prediction of sentiment represented as float array for a given string input<br>
Prediction pipeline consists of three main parts
- preprocessing pipeline
- postprocessing pipeline
- embedding and prediction

#### Preprocessing pipeline

Applies all preprocessing steps on string input
- removing links
- removing usernames
- removing punctuation
- lowering all characters
- tokenization

#### Postprocessing pipeline
Uses tokens from preprocessing pipeline and **removes stopwords** and **lemmatizes** them

#### Embedding and prediction
Embeds tokens using _Spacy_ library and uses trained model to return sentiment prediction

# Links
- Dataset - https://www.kaggle.com/kazanova/sentiment140
- Keras library - https://keras.io/
- Scikit-learn library - https://scikit-learn.org/stable/
- Spacy library - https://spacy.io/
- NLTK library - https://www.nltk.org/
- Flask library - https://palletsprojects.com/p/flask/
