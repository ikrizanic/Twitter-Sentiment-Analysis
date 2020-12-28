import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


class Postprocessing:
    """
    Methods:
        remove_stopwords(list): returns list of tokens without stopwords
        lemmatize(list): returns lemmas of words from the tokens list
        embed(list): returns vectors made with spacy
        embed_all(list): returns vectors for all data
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('wordnet')

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def postprocessing_pipeline(self, raw, group=False):
        """
        Parameters:
            raw: list of input strings or single string
            group: if True, works with list of input data, else expects single string input
        """
        if group:
            return [self.lemmatize(self.remove_stopwords(tweet)) for tweet in raw]

        return self.lemmatize(self.remove_stopwords(raw))
