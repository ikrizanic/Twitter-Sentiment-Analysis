from src.Embedding import Embedding
from src.Postprocessing import Postprocessing
from src.Preprocessing import Preprocessing


class PredictionPipeline:

    def __init__(self, vocab, model):
        self.preprocessing = Preprocessing()
        self.postprocessing = Postprocessing()
        self.embedding = Embedding()
        self.model = model
        self.vocab = vocab

    def get_prediction(self, text):
        data = self.postprocessing.postprocessing_pipeline(self.preprocessing.preprocessing_pipeline(text))
        x = self.embedding.encode_data(data, vocab=self.vocab, single=True)
        return self.model.predict(x)
