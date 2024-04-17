"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf

import os

from apputils.utils import multiclass_binning
from providers.df import df_clean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 50


def scheduler(epoch, lr):
    if epoch < 5:
        return round(lr, 4)
    else:
        return round(lr * math.exp(-0.1), 4)


def tf_learn(raw_data, pct, is_multiclass, features=None):
    evaluations = dict()
    if features is not None:
        features_vals = features.value
        frame_cols = raw_data.columns.tolist()
        criteria_list = [x for x in frame_cols if x in features_vals]
        raw_data = raw_data[criteria_list]
    if not is_multiclass:
        try:
            ds_len = len(raw_data)
            pct_len = int(round(pct * ds_len, 0))
            raw_data=raw_data.sample(frac=1)
            train_pd_ds = raw_data.head(ds_len - pct_len)
            test_pd_ds = raw_data.tail(pct_len)
            train_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd_ds, label='estimated',max_num_classes=2)
            test_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(test_pd_ds, label='estimated',max_num_classes=2)
            tuner = tfdf.tuner.RandomSearch(num_trials=40,use_predefined_hps=True,trial_num_threads=10)
            model_rf = tfdf.keras.RandomForestModel(tuner=tuner,verbose=2)
            model_rf.compile(metrics=["accuracy"])
            model_rf.fit(train_ds_tf)
            model_rf.make_inspector().evaluation()
            evaluate_rf = model_rf.evaluate(test_ds_tf, return_dict=True)
        except Exception as ex:
            print(ex.__str__())
        lr_scheduler_multiclass = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_stop_mc = ks.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      mode='max', min_delta=0.001,
                                                      patience=5)

        ds = raw_data
        labels = ["less or equal", "greater"]
        df_X = ds.drop(["estimated"], axis=1)
        df_Y = ds["estimated"].values
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=pct,
                                                            shuffle=True, random_state=42)
        input_nodes = int(df_X.shape[1])
        output_nodes = len(labels)
        hidden_units = 8
        scaler = MinMaxScaler()
        Xtr_scaled = scaler.fit_transform(X_train)
        Xts_scaled = scaler.fit_transform(X_test)
        try:
            model_multiclass = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units * 2), activation='relu', kernel_initializer="he_normal"),
                ks.layers.Dense(int(hidden_units / 2), activation='relu', kernel_initializer="he_normal"),
                ks.layers.Dense(output_nodes, activation='sigmoid')])
            model_multiclass.compile(optimizer="adam",
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=['accuracy'])
            model_multiclass.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                                 validation_data=[Xts_scaled, Y_test],
                                 callbacks=[lr_scheduler_multiclass, ])  # callback_stop_mc])
            evaluate_multiclass = model_multiclass.evaluate(Xts_scaled, Y_test)
            evaluations['classifiers'] = {
                'loss': evaluate_multiclass[0],
                'accuracy': evaluate_multiclass[1]
            }
        except Exception as ex:
            print(ex.__str__())
    else:
        lr_scheduler_multiclass = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
        lr_scheduler_regressor = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_stop_mc = ks.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      mode='max', min_delta=0.001,
                                                      patience=8)
        callback_stop_rg = ks.callbacks.EarlyStopping(monitor='val_mae',
                                                      mode='max', min_delta=0.001,
                                                      patience=5)

        ds = raw_data[0]
        labels = raw_data[2]
        df_X = ds.drop(["estimated"], axis=1)
        df_Y = ds["estimated"].values
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=pct,
                                                            shuffle=True, random_state=42)
        input_nodes = int(df_X.shape[1])
        output_nodes = len(labels)
        hidden_units = 32
        scaler = MinMaxScaler()
        Xtr_scaled = scaler.fit_transform(X_train)
        Xts_scaled = scaler.fit_transform(X_test)
        try:
            model_multiclass = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units / 2), activation='relu', kernel_initializer="he_normal"),
                ks.layers.Dense(int(hidden_units / 4), activation='relu'),
                ks.layers.Dense(int(hidden_units / 8), activation='relu'),
                ks.layers.Dense(output_nodes, activation='softmax')])
            model_multiclass.compile(optimizer="adam",
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=['accuracy'])
            model_multiclass.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                                 validation_data=[Xts_scaled, Y_test]
                                 , callbacks=[lr_scheduler_multiclass, callback_stop_mc])
            evaluate_multiclass = model_multiclass.evaluate(Xts_scaled, Y_test)
            evaluations['classifiers'] = {
                'loss': evaluate_multiclass[0],
                'accuracy': evaluate_multiclass[1]
            }
        except Exception as ex:
            print(ex.__str__())
        try:
            model_regress = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units / 2), activation='leakyrelu', kernel_initializer="he_normal"),
                ks.layers.Dense(int(hidden_units / 4), activation='leakyrelu'),
                ks.layers.Dense(int(hidden_units / 8), activation='relu'),
                ks.layers.Dense(output_nodes, activation='sigmoid')])
            model_regress.compile(optimizer="adam", loss='mse', metrics=['mae'])
            model_regress.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                              validation_data=[Xts_scaled, Y_test],
                              callbacks=[lr_scheduler_regressor, callback_stop_rg])
            evaluate_regress = model_regress.evaluate(Xts_scaled, Y_test)
            evaluations['regressors'] = {
                'loss:MSE': evaluate_regress[0],
                'accuracy:MAE': evaluate_regress[1]
            }
        except Exception as ex:
            print(ex.__str__())

    pass
