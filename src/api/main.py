import pickle

from api.Utils import one_hot_encoder
from api.Embedding import Embedding
from api.Utils import load_dataset
from api.Preprocessing import Preprocessing
from api.Postprocessing import Postprocessing
import pandas as pd
from PredictionPipeline import PredictionPipeline
from tensorflow import keras


def main():
    # pd.options.mode.chained_assignment = None
    #
    # prep = Preprocessing()
    # post = Postprocessing()
    # emb = Embedding()

    # print("Loading data...")
    # df = load_dataset("../dataset/dataset.csv")

    # labels = one_hot_encoder(df['label'])

    # print("Preprocessing...")
    # df['text'] = prep.preprocessing_pipeline(df['text'], group=True)
    #
    # print("Postprocessing...")
    # df['text'] = post.postprocessing_pipeline(df['text'], group=True)
    #
    # # vocab = emb.build_vocab(df['text'], size=50000)
    with open("../../data/vocab.pl", "rb") as f:
        # pickle.dump(vocab, f)
        vocab = pickle.load(f)
    #
    # # embedding_matrix = emb.make_embedding_matrix(vocab, embedding_dim=300)
    # with open("../data/embedding_matrix.pl", "rb") as f:
    #     # pickle.dump(embedding_matrix, f)
    #     embedding_matrix = pickle.load(f)
    #
    # print("Embedding...")
    # # encoded_data = emb.encode_data(df, vocab)['text']
    # with open("../data/encoded_data.pl", "rb") as f:
    #     # pickle.dump(encoded_data, f)
    #     encoded_data = pickle.load(f)
    #
    # print("Padding...")
    # size = min(15, max(len(e) for e in encoded_data))
    # data = emb.pad_encoded_data(encoded_data, size=size)
    #
    # x_train, x_test, y_train, y_test = get_data_splits(data, labels, test_size=0.2)
    #
    # x_val, x_test, y_val, y_test = get_data_splits(x_test, y_test, test_size=0.5)
    #
    # api = keras.Sequential()
    # # Add an Embedding layer expecting input vocab of size 1000, and
    # # output embedding dimension of size 64.
    # api.add(layers.Embedding(input_dim=len(vocab) + 1, output_dim=300, weights=[embedding_matrix],
    #                            trainable=False, mask_zero=True))
    #
    # # Masking layer
    # api.add(Masking(mask_value=0.0))
    #
    # # Add a LSTM layer with 128 internal units.
    # api.add(layers.LSTM(128, activation='relu'))
    #
    # # Add a Dense layer with 2 units.
    # api.add(layers.Dense(2, activation='softmax'))
    #
    # api.summary()
    # api.compile(optimizer='adam', loss=keras.losses.SquaredHinge(reduction="auto", name="squared_hinge"),
    #               metrics=[keras.metrics.Recall()])
    # callbacks = [EarlyStopping(monitor='val_loss', patience=3),
    #              ModelCheckpoint("/home/ikrizanic/Documents/git_repos/Twitter-Sentiment-Analysis/api.h5")]
    #
    # def fit_model(api, X_train, y_train, X_val, y_val, batch_size=128, epochs=200):
    #     history = api.fit(X_train, y_train,
    #                         batch_size=batch_size, epochs=epochs,
    #                         callbacks=callbacks,
    #                         validation_data=(X_val, y_val))
    #     return history, api
    #
    # def evaluate_model(api, X_test, y_test):
    #     res = api.evaluate(X_test, y_test)
    #     return res
    #
    # history, model = fit_model(api, x_train, y_train, x_val, y_val, batch_size=2048, epochs=2)
    model = keras.models.load_model("../../saved_models/")
    pp = PredictionPipeline(vocab, model)
    print(pp.get_prediction("This is very nice Tweet and I'm happy!"))
    print(pp.get_prediction("I hate myself"))
    print(pp.get_prediction("GO TO HELL"))
    print(pp.get_prediction("Life is good!!!"))
    print(pp.get_prediction("I'm gonna find you and im gonna kill you :)"))
    print("Done!")


if __name__ == '__main__':
    main()
