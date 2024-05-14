import tensorflow as tf
from keras import Model,Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow import keras as ks


def create_baseline_model(hidden_units, features, output):
    inputs = ks.layers.Input((len(features),))
    flats=ks.layers.Flatten()(inputs)
    norms= ks.layers.BatchNormalization()(flats)
    for units in hidden_units:
        norms = ks.layers.Dense(units, activation="leaky_relu",kernel_initializer='he_normal')(norms)
    # The output is deterministic: a single point estimate
    dropout=ks.layers.Dropout(0.2)(norms)
    outputs = ks.layers.Dense(units=output, activation="sigmoid")(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_lstm_model(hidden_units,features,output,ds,internal_activation,result_activation):
    EMBEDDING_DIM=len(features)
    model=Sequential()
    model.add(ks.layers.Input((EMBEDDING_DIM,)))
    model.add(Embedding(len(ds),EMBEDDING_DIM))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(ks.layers.Dropout(0.2))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(ks.layers.Dropout(0.2))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(ks.layers.Dropout(0.2))
    model.add(LSTM(units=256,activation="leaky_relu"))
    model.add(ks.layers.Dropout(0.2))
    model.add(Dense(units=1))
    model.add(ks.layers.Dropout(0.2))
    model.add(Dense(units=1,activation='sigmoid'))
 #   embedding_layer=Embedding(len(ds),EMBEDDING_DIM)(inputs)
 #   lstm_layer_1=LSTM(hidden_units,activation=internal_activation,return_sequences=True)(embedding_layer)
 #   bidi=ks.layers.Bidirectional(LSTM(hidden_units,activation=internal_activation))(lstm_layer_1)
 #   outputs=ks.layers.Dropout(0.2)(bidi)
 #   outputs=Dense(int(hidden_units/4),activation=internal_activation)(outputs)
 #   outputs=ks.layers.Dense(output,activation=result_activation,kernel_initializer="he_normal")(outputs)
 #model=Model(inputs,outputs)
    return model
