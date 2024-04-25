"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math
import numpy as np
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import os

from apputils.utils import prepare_con_mat
from providers.df import df_remove_outliers
from providers.ui import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001 * 10 ** (epoch / 20)


def tf_learn_model(inp_ds, pct, is_multiclass, features=None, seasoning=False, skiptree=False):
    raw_data = df_remove_outliers(inp_ds.copy(deep=True)) if not is_multiclass else inp_ds[0].copy(deep=True)
    evaluations = dict()
    ds_len = len(raw_data)
    pct_len = int(round(pct * ds_len, 0))
    num_train_examples = 6000
    num_epochs = 50
    batch_size = 8
    batch_size_trees = 128

    if features is not None:
        raw_data = raw_data if not is_multiclass else raw_data[0]
        features_vals = features.value
        frame_cols = raw_data.columns.tolist()
        criteria_list = [x for x in frame_cols if x in features_vals]
        raw_data = raw_data[criteria_list]
    train_pd_ds = raw_data.head(ds_len - pct_len)
    features: list = train_pd_ds.columns.tolist()
    features.remove('estimated')
    test_pd_ds = raw_data.tail(pct_len)
    tf_train = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(train_pd_ds[features].values, tf.float32),
                tf.cast(train_pd_ds['estimated'].values, tf.int32)
            )
        )
    )
    tf_test = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(test_pd_ds[features].values, tf.float32),
                tf.cast(test_pd_ds['estimated'].values, tf.int32)
            )
        )
    )
    if not is_multiclass:

        if not skiptree:
            try:
                tf_train_trees = tf_train.batch(batch_size_trees)
                tf_test_trees = tf_test.batch(batch_size_trees)
                tuner = tfdf.tuner.RandomSearch(num_trials=200,
                                                trial_num_threads=12)
                model_rf = tfdf.keras.RandomForestModel(tuner=tuner, )
                model_hboost = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, )
                model_rf.fit(tf_train_trees)
                model_hboost.fit(tf_train_trees)
                model_hboost.compile(metrics=['accuracy'])
                model_rf.compile(metrics=["accuracy"])
                evaluate_rf = model_rf.evaluate(tf_test_trees, return_dict=True)
                evaluate_hboost = model_hboost.evaluate(tf_test_trees, return_dict=True)
                predict_rf = model_rf.predict(test_pd_ds[features].values)
                predict_hboost = model_hboost.predict(test_pd_ds[features].values)
                confmx_rf = tf.math.confusion_matrix(predict_rf, test_pd_ds['estimated'].values)
                print(confmx_rf)
                confmx_hboost = tf.math.confusion_matrix(predict_hboost, test_pd_ds['estimated'].values)
                print(confmx_hboost)
                do_plot_train_trees(model_rf, "RandomForestClassifier binary classification")
                do_plot_train_trees(model_hboost, "GradientBoostTrees binary classification")
            # do_plot_conf_mx(df_mx_hboost,"Confusion matrix GradientBoostTrees binary classification")
            except Exception as ex:
                print(ex.__str__())
        try:
            tf_train_nn = tf_train.shuffle(
                num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
            tf_test_nn = tf_test.shuffle(
                num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
            lr_scheduler_multiclass = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
            callback_stop_mc = ks.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                                          mode='max', min_delta=0.001,
                                                          patience=8)
            model_class = ks.Sequential([
                ks.layers.Flatten(input_shape=(len(features),)),
                ks.layers.BatchNormalization(trainable=False),
                ks.layers.Dense(4095, activation='leaky_relu', kernel_initializer="uniform"),
                ks.layers.Dropout(0.1),
                ks.layers.Dense(2047, activation='linear', kernel_initializer="uniform"),
                ks.layers.BatchNormalization(trainable=False),
                ks.layers.Dense(15, activation='leaky_relu'),
                ks.layers.Dense(7, activation='linear', kernel_initializer="uniform"),
                ks.layers.BatchNormalization(trainable=False),
                ks.layers.Dense(5, activation='leaky_relu'),
                ks.layers.Dense(1, activation='sigmoid', kernel_initializer="uniform"),
            ])
            model_class.compile(optimizer=ks.optimizers.Adam(learning_rate=0.005, amsgrad=True),
                                loss='binary_crossentropy',
                                metrics=['binary_accuracy'])
            history = model_class.fit(tf_train_nn, epochs=num_epochs, validation_data=tf_test_nn,
                                      shuffle=True, steps_per_epoch=math.ceil(num_train_examples / batch_size),
                                      callbacks=[callback_stop_mc])
            evaluate_class = model_class.evaluate(tf_test_nn)
            evaluations['classifiers'] = {
                'loss': evaluate_class[0],
                'accuracy': evaluate_class[1]
            }
            do_plot_history_seq(history, "Sequential NN binary classification", "binary_accuracy")
        except Exception as ex:
            print(ex.__str__())
    else:
        batch_size = 1
        lr_scheduler_multiclass = ks.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callback_stop_mc = ks.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      mode='max', min_delta=0.001,
                                                      patience=8)
        tf_train_nn = tf_train.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        tf_test_nn = tf_test.shuffle(
            num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
        input_nodes = int(len(features))
        output_nodes = len(inp_ds[2])
        try:
            model_multiclass = ks.Sequential([
                ks.layers.Flatten(input_shape=(input_nodes,)),
                ks.layers.BatchNormalization(trainable=False),
                ks.layers.Dense(2048, activation='leaky_relu', kernel_initializer="uniform"),
                ks.layers.Dropout(0.3),
                ks.layers.Dense(512, activation='softmax', kernel_initializer="uniform"),
                ks.layers.BatchNormalization(trainable=False),
                ks.layers.Dense(15, activation='leaky_relu'),
                ks.layers.Dense(output_nodes, activation='sigmoid', kernel_initializer="uniform"),
            ])
            model_multiclass.compile(ks.optimizers.Adam(learning_rate=0.005, amsgrad=True), loss="mse", metrics=['mae'])
            history = model_multiclass.fit(tf_train_nn, batch_size=batch_size, epochs=num_epochs,
                                           validation_data=tf_test_nn,
                                           shuffle=True, steps_per_epoch=math.ceil(num_train_examples / batch_size)
                                           , callbacks=[callback_stop_mc])
            evaluate_multiclass = model_multiclass.evaluate(tf_test_nn)
            evaluations['regression'] = {
                'loss': evaluate_multiclass[0],
                'mse': evaluate_multiclass[1]
            }
            do_plot_history_seq(history, "Sequential NN muliclass classification", metric='mse')
        except Exception as ex:
            print(ex.__str__())
