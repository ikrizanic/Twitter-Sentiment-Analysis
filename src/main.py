from src.Utils import load_dataset, save_embeddings
from src.Preprocessing import Preprocessing
from src.Postprocessing import Postprocessing


def main():
    prep = Preprocessing()
    post = Postprocessing(MAX_NUMBER_OF_VECTORS=30)

    df = load_dataset("../dataset/dataset.csv")

    df['text'] = prep.preprocessing_pipe(df['text'], group=True)
    embeddings = post.embed_all(df['text'])
    with open("../data/embeddings.pl", 'wb') as f:
        save_embeddings(embeddings, f)


if __name__ == '__main__':
    main()
