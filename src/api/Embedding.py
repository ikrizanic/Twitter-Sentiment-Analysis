import spacy
from tqdm import tqdm
import numpy as np
from collections import Counter


class Embedding:
    def __init__(self, MAX_NUMBER_OF_VECTORS=30):
        self.nlp = spacy.load("en_core_web_lg", disable=['tokenizer', 'parser', 'ner', 'tagger'])
        self.MAX_NUMBER_OF_VECTORS = MAX_NUMBER_OF_VECTORS
        self.word_counter = Counter()

    def build_vocab(self, data, min_counts=10, size=None):
        for d in data:
            self.word_counter.update(d)
        my_vocab = dict()
        index = 1
        if size is None:
            for k, v in self.word_counter.items():
                if v >= min_counts:
                    my_vocab.update({k: index})
                    index += 1
        else:
            for k, v in sorted(self.word_counter.items(), key=lambda item: item[1], reverse=True):
                if len(my_vocab) < size:
                    my_vocab.update({k: index})
                    index += 1
                else:
                    break

        return my_vocab

    def encode_data(self, data, vocab, single=False):
        if single:
            return self.pad_encoded_data([[vocab.get(token, 0) for token in data]])
        for i in tqdm(range(len(data['text']))):
            data['text'][i] = [vocab.get(token, 0) for token in data['text'][i]]
        return data

    def make_embedding_matrix(self, vocab, embedding_dim=300):
        hits, misses = 0, 0
        embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
        for word, i in vocab.items():
            token = self.nlp(word)
            if token.has_vector:
                embedding_matrix[i] = token.vector
                hits += 1
            else:
                misses += 1

        print("Converted %d words (%d misses)" % (hits, misses))
        return embedding_matrix

    def pad_encoded_data(self, encoded, size=None):
        if size is None:
            size = self.MAX_NUMBER_OF_VECTORS
        features = np.zeros((len(encoded), size), dtype=float)
        for i, review in enumerate(encoded):
            if len(review) > size:
                review = review[:size]
            zeroes = list(np.zeros(size - len(review)))
            new = zeroes + review
            features[i, :] = np.array(new)
        return features
