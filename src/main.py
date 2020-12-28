import pickle
from src.Embedding import Embedding
from src.Utils import load_dataset
from src.Preprocessing import Preprocessing
from src.Postprocessing import Postprocessing
import pandas as pd


def main():
    pd.options.mode.chained_assignment = None

    prep = Preprocessing()
    post = Postprocessing()
    emb = Embedding()

    print("Loading data...")
    df = load_dataset("../dataset/dataset.csv")

    print("Preprocessing...")
    df['text'] = prep.preprocessing_pipeline(df['text'], group=True)

    print("Postprocessing...")
    df['text'] = post.postprocessing_pipeline(df['text'], group=True)

    # vocab = emb.build_vocab(df['text'], size=50000)
    with open("../data/vocab.pl", "rb") as f:
        # pickle.dump(vocab, f)
        vocab = pickle.load(f)

    # embedding_matrix = emb.make_embedding_matrix(vocab, embedding_dim=300)
    with open("../data/embedding_matrix.pl", "rb") as f:
        # pickle.dump(embedding_matrix, f)
        embedding_matrix = pickle.load(f)

    print("Embedding...")
    encoded_data = emb.encode_data(df, vocab)['text']
    with open("../data/encoded_data.pl", "wb") as f:
        pickle.dump(encoded_data, f)


if __name__ == '__main__':
    main()
