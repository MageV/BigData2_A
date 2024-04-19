"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math

import keras.random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, SplineTransformer
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf

import os

from apputils.utils import multiclass_binning
from providers.df import df_clean
from providers.seasonal import preprocess_seasonal
from providers.ui import do_plot_train_trees, do_plot_history_seq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 50


def scheduler(epoch, lr):
    if epoch < 5:
        return round(lr, 4)
    else:
        return round(lr * math.exp(-0.1), 4)


def custom_layer_initializer(shape, dtype=None, prob=0):
    return ks.backend.binomial(shape=shape, dtype=dtype, p=prob)


def tf_learn_model(raw_data, pct, is_multiclass, features=None, seasoning=False):
    evaluations = dict()
    ds_len = len(raw_data)
    pct_len = int(round(pct * ds_len, 0))

    if features is not None:
        raw_data = raw_data if not is_multiclass else raw_data[0]
        features_vals = features.value
        frame_cols = raw_data.columns.tolist()
        criteria_list = [x for x in frame_cols if x in features_vals]
        raw_data = raw_data[criteria_list]

    if not is_multiclass:
        try:
            train_pd_ds = raw_data.head(ds_len - pct_len).sample(frac=1)
            test_pd_ds = raw_data.tail(pct_len).sample(frac=1)
            train_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd_ds, label='estimated',
                                                                task=tfdf.keras.Task.CLASSIFICATION)
            test_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(test_pd_ds, label='estimated',
                                                               task=tfdf.keras.Task.CLASSIFICATION)
            tuner = tfdf.tuner.RandomSearch(num_trials=100, use_predefined_hps=True, trial_num_threads=12)
            model_rf = tfdf.keras.RandomForestModel(tuner=tuner, verbose=2)
            model_hboost = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, verbose=2)
            model_rf.fit(train_ds_tf)
            model_hboost.fit(train_ds_tf)
            model_hboost.compile(metrics=['accuracy'])
            model_rf.compile(metrics=["accuracy"])
            evaluate_rf = model_rf.evaluate(test_ds_tf, return_dict=True)
            evaluate_hboost = model_hboost.evaluate(test_ds_tf, return_dict=True)
            do_plot_train_trees(model_rf)
            do_plot_train_trees(model_hboost)
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
        #    scaler = MinMaxScaler()
        #    Xtr_scaled = scaler.fit_transform(X_train)
        #    Xts_scaled = scaler.fit_transform(X_test)
        try:
            model_class = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units), activation='relu',
                                kernel_initializer=custom_layer_initializer),
                ks.layers.Dropout(0.5),
                ks.layers.Dense(int(hidden_units / 2), activation='relu'),
                ks.layers.Dropout(0.5),
                ks.layers.Dense(output_nodes, activation='softmax')])
            model_class.compile(optimizer="adam",
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])
            model_class.fit(X_train, Y_train, batch_size=1, epochs=EPOCHS,
                            validation_data=[X_test, Y_test], shuffle=True,
                            callbacks=[lr_scheduler_multiclass, callback_stop_mc])
            evaluate_class = model_class.evaluate(X_test, Y_test)
            evaluations['classifiers'] = {
                'loss': evaluate_class[0],
                'accuracy': evaluate_class[1]
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
        scaler = SplineTransformer(degree=4, n_knots=8,knots='quantile',extrapolation='periodic',include_bias=True)
        #        Xtr_scaled = ks.utils.normalize(X_train, axis=1)
        #        Xts_scaled = ks.utils.normalize(X_test, axis=1)
        #scaler = MinMaxScaler()
        Xtr_scaled = scaler.fit_transform(X_train)
        Xts_scaled = scaler.fit_transform(X_test)
        try:
            model_multiclass = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units / 2), activation='relu',
                                kernel_initializer=custom_layer_initializer),
                ks.layers.Dense(int(hidden_units / 4), activation='relu'),
                ks.layers.Dense(int(hidden_units / 8), activation='relu'),
                ks.layers.Dense(output_nodes, activation='softmax')])
            model_multiclass.compile(optimizer="adam",
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                     metrics=['accuracy'])
            history=model_multiclass.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                                 validation_data=[Xts_scaled, Y_test]
                                 , callbacks=[lr_scheduler_multiclass, callback_stop_mc])
            evaluate_multiclass = model_multiclass.evaluate(Xts_scaled, Y_test)
            evaluations['classifiers'] = {
                'loss': evaluate_multiclass[0],
                'accuracy': evaluate_multiclass[1]
            }
            do_plot_history_seq(history)
        except Exception as ex:
            print(ex.__str__())
        try:
            model_regress = ks.Sequential([
                ks.layers.InputLayer((input_nodes,)),
                ks.layers.Dense(int(hidden_units / 2), activation='leaky_relu',
                                kernel_initializer=custom_layer_initializer),
                ks.layers.Dense(int(hidden_units / 4), activation='leaky_relu'),
                ks.layers.Dense(int(hidden_units / 8), activation='relu'),
                ks.layers.Dense(output_nodes, activation='sigmoid')])
            model_regress.compile(optimizer="adam", loss='mse', metrics=['mae'])
            history=model_regress.fit(Xtr_scaled, Y_train, batch_size=1, epochs=EPOCHS,
                              validation_data=[Xts_scaled, Y_test],
                              callbacks=[lr_scheduler_regressor, callback_stop_rg])
            evaluate_regress = model_regress.evaluate(Xts_scaled, Y_test)
            evaluations['regressors'] = {
                'loss:MSE': evaluate_regress[0],
                'accuracy:MAE': evaluate_regress[1]
            }
            do_plot_history_seq(history)
        except Exception as ex:
            print(ex.__str__())
        try:
            ds = raw_data[0]
            ds_len = len(ds)
            pct_len = int(round(pct * ds_len, 0))
            train_pd_ds = ds.head(ds_len - pct_len).sample(frac=1)
            test_pd_ds = ds.tail(pct_len).sample(frac=1)
            train_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd_ds, label='estimated',
                                                                task=tfdf.keras.Task.REGRESSION)
            test_ds_tf = tfdf.keras.pd_dataframe_to_tf_dataset(test_pd_ds, label='estimated',
                                                               task=tfdf.keras.Task.REGRESSION)
            tuner = tfdf.tuner.RandomSearch(num_trials=100, use_predefined_hps=True, trial_num_threads=12)
            model_rf = tfdf.keras.RandomForestModel(tuner=tuner, verbose=2, task=tfdf.keras.Task.REGRESSION)
            model_hboost = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, verbose=2, task=tfdf.keras.Task.REGRESSION)
            model_cart = tfdf.keras.CartModel(tuner=tuner, verbose=2, task=tfdf.keras.Task.REGRESSION)
            model_rf.fit(train_ds_tf)
            model_hboost.fit(train_ds_tf)
            model_hboost.compile(metrics=['accuracy'])
            model_rf.compile(metrics=["accuracy"])
            model_cart.compile(metrics=["accuracy"])
            evaluate_rf = model_rf.evaluate(test_ds_tf, return_dict=True)
            evaluate_hboost = model_hboost.evaluate(test_ds_tf, return_dict=True)
            evaluate_cart = model_cart.evaluate(test_ds_tf, return_dict=True)
            do_plot_train_trees(model_rf)
            do_plot_train_trees(model_hboost)
            do_plot_train_trees(model_cart)
        except Exception as ex:
            print(ex.__str__())

    pass
