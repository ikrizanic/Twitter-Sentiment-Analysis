import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical


def load_dataset(path, encoding="ISO-8859-1", names=None):
    if names is None:
        names = ["label", "id", "date", "flag", "user", "text"]
    return pd.read_csv(path, encoding=encoding, names=names)


def get_data_splits(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size)


def one_hot_encoder(labels):
    new_labels = normalize_labels(labels)
    return to_categorical(new_labels)


def normalize_labels(labels):
    return [0 if l == 0 else 1 for l in labels]


