import tensorflow as tf
from keras import Model, Sequential
from keras.src.initializers import VarianceScaling
from keras.src.layers import Bidirectional
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow import keras as ks


def create_baseline_model_binary(hidden_units, features, output):
    inputs = ks.layers.Input((len(features),))
    flats = ks.layers.Flatten()(inputs)
    norms = ks.layers.BatchNormalization()(flats)
    for units in hidden_units:
        norms_1 = ks.layers.Dense(units, activation="leaky_relu", kernel_initializer="he_normal")(norms)
        norms_2 = ks.layers.Dense(int(units / 2), activation="relu", kernel_initializer="he_normal")(norms)
        norms = ks.layers.Concatenate(axis=1)([norms_1, norms_2])
    dropout = ks.layers.Dropout(0.2)(norms)
    outputs = ks.layers.Dense(units=output, activation="sigmoid", kernel_initializer="normal")(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_baseline_model_multi(hidden_units, features, output):
    model = Sequential()
    model.add(ks.layers.Input((len(features),)))
    model.add(ks.layers.BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(ks.layers.BatchNormalization())
    model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.Flatten())
    model.add(Dense(output, activation='sigmoid'))
    return model


def create_lstm_model(features, output, ds):
    EMBEDDING_DIM = len(features)
    model = Sequential()
    model.add(ks.layers.Input((EMBEDDING_DIM,)))
    model.add(Embedding(len(ds), EMBEDDING_DIM))
    model.add(ks.layers.BatchNormalization())
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model.add(LSTM(units=256))
    model.add(Dense(units=256, kernel_initializer="he_normal",activation="leaky_relu"))
    model.add(ks.layers.Flatten())
    model.add(Dense(units=1, activation='sigmoid'))

    return model
