from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm


class Postprocessing:
    """
    Methods:
        remove_stopwords(list): returns list of tokens without stopwords
        lemmatize(list): returns lemmas of words from the tokens list
        embed(list): returns vectors made with spacy
        embed_all(list): returns vectors for all data
    """

    def __init__(self, MAX_NUMBER_OF_VECTORS=30):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_lg", disable=['tokenizer', 'parser', 'ner', 'tagger'])
        self.MAX_NUMBER_OF_VECTORS = MAX_NUMBER_OF_VECTORS

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def embed(self, tokens):
        return [self.nlp(token).vector for token in tokens][:self.MAX_NUMBER_OF_VECTORS]

    def embed_all(self, data):
        return [self.embed(data[i]) for i in tqdm(range(len(data)))]


