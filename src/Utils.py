import pandas as pd
import pickle


def load_dataset(path, encoding="ISO-8859-1", names=None):
    if names is None:
        names = ["label", "id", "date", "flag", "user", "text"]
    return pd.read_csv(path, encoding=encoding, names=names)


def save_embeddings(embeddings, path):
    pickle.dump(embeddings, path)
