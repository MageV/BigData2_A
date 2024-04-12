"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split

import os

from tensorflow.python.keras.initializers.initializers_v2 import LecunNormal

from apputils.utils import multiclass_binning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 50


def scheduler(epoch, lr):
    if epoch < 5:
        return round(lr, 4)
    else:
        return round(lr * math.exp(-0.1), 4)


def tf_learn(dataset, drop, classified_by, pct):
    evaluations = dict()
    lr_scheduler_multiclass = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
    lr_scheduler_regressor = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
    callback_stop_mc = ks.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  mode='max', min_delta=0.001,
                                                  patience=5)
    callback_stop_rg = ks.callbacks.EarlyStopping(monitor='val_mae',
                                                  mode='max', min_delta=0.001,
                                                  patience=5)
    raw_data, boundaries, labels = multiclass_binning(dataset, classified_by, 8)
    raw_data.drop([drop, classified_by], axis=1, inplace=True)
    raw_data.dropna(inplace=True)
    df_X = raw_data.drop(["estimated"], axis=1)
    df_Y = raw_data["estimated"].values
    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=pct,
                                                        shuffle=True, random_state=42)
    input_nodes = int(df_X.shape[1])
    output_nodes = len(labels)
    hidden_units = 16
    scaler = MinMaxScaler()
    Xtr_scaled = scaler.fit_transform(X_train)
    Xts_scaled = scaler.fit_transform(X_test)
    try:
        model_multiclass = ks.Sequential([
            ks.layers.InputLayer((input_nodes,), activation='relu', kernel_initializer="he_normal"),
            ks.layers.Dense(int(hidden_units / 2), activation='selu', kernel_initializer=LecunNormal()),
            ks.layers.Dense(int(hidden_units / 4), activation='selu', kernel_initializer=LecunNormal()),
            ks.layers.Dense(int(hidden_units / 8), activation='relu'),
            ks.layers.Dense(output_nodes, activation='softmax')])
        model_multiclass.compile(optimizer="adam",
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])
        model_multiclass.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                             validation_data=[Xts_scaled, Y_test],
                             callbacks=[lr_scheduler_multiclass, callback_stop_mc])
        evaluate_multiclass = model_multiclass.evaluate(Xts_scaled, Y_test)
        evaluations['classifiers'] = {
            'loss': evaluate_multiclass[0],
            'accuracy': evaluate_multiclass[1]
        }
    except Exception as ex:
        print(ex.__str__())
    try:
        model_regress = ks.Sequential([
            ks.layers.Dense(input_nodes, activation='selu', kernel_initializer=LecunNormal()),
            ks.layers.Dense(hidden_units, activation='selu', kernel_initializer=LecunNormal()),
            ks.layers.Dense(int(hidden_units / 2), activation='relu'),
            ks.layers.Dense(output_nodes, activation='sigmoid')])
        model_regress.compile(optimizer="sgd", loss='mse', metrics=['mae'])
        model_regress.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                          validation_data=[Xts_scaled, Y_test],
                          callbacks=[lr_scheduler_regressor, callback_stop_mc])
        evaluate_regress = model_regress.evaluate(Xts_scaled, Y_test)
        evaluations['regressors'] = {
            'loss:MSE': evaluate_regress[0],
            'accuracy:MAE': evaluate_regress[1]
        }
    except Exception as ex:
        print(ex.__str__())
    print(evaluations)
    pass
