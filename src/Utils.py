import pandas as pd


def load_dataset(path, encoding="ISO-8859-1", names=["label", "id", "date", "flag", "user", "text"]):
    return pd.read_csv(path, encoding=encoding, names=names)
