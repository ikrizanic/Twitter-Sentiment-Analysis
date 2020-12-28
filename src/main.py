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

    df = load_dataset("../dataset/dataset.csv")

    df['text'] = prep.preprocessing_pipeline(df['text'], group=True)
    df['text'] = post.postprocessing_pipeline(df['text'], group=True)

    vocab = emb.build_vocab(df['text'], size=50000)
    print(len(vocab))
    with open("../data/vocab.pl", "wb") as f:
        pickle.dump(vocab, f)

    encoded_data = emb.pad_encoded_data(emb.encode_data(df, vocab)['text'])
    with open("../data/encoded_data.pl", "wb") as f:
        pickle.dump(encoded_data, f)

    embedding_matrix = emb.make_embedding_matrix(vocab, embedding_dim=300)
    with open("../data/embedding_matrix.pl", "wb") as f:
        pickle.dump(embedding_matrix, f)


if __name__ == '__main__':
    main()
