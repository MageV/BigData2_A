"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.model_selection._split import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os

from apputils.utils import multiclass_binning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_learn(dataset, drop, classified_by, pct):
    raw_data, boundaries, labels = multiclass_binning(dataset, classified_by)
    source_data = raw_data.drop([drop,classified_by], axis=1)
    df_X = source_data.drop(['estimated'], axis=1)
    df_Y = source_data['estimated'].values.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=pct,
                                                        shuffle=True, random_state=42, stratify=df_Y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model = Sequential()
    model.add(LSTM(X_train_scaled.shape[1], return_sequences=True, input_shape=[X_train_scaled.shape[1], 1]))
    model.add(LSTM(X_train.shape[1], return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=1, epochs=30)
    Y_pred = model.predict(X_test_scaled)
    metrics = model.compute_metrics(X_test_scaled, Y_test, Y_pred)
    pass
