"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.layers import Concatenate, Subtract
from keras.src.utils import plot_model

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import os

from apputils.utils import prepare_con_mat
from providers.df import df_remove_outliers
from providers.ui import *
from config.appconfig import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001 * 10 ** (epoch / 20)


def tf_learn_model(inp_ds, pct, is_multiclass, features=None, skiptree=False, tface=None):
    raw_data = df_remove_outliers(inp_ds.copy(deep=True)) if not is_multiclass else inp_ds[0].copy(deep=True)
    if tface is not None:
        raw_data = raw_data.loc[raw_data['typeface'] == tface.value]
        raw_data.drop('typeface', inplace=True,axis=1)
    evaluations = dict()
    ds_len = len(raw_data)
    pct_len = int(round(pct * ds_len, 0))
    num_train_examples = 6000
    num_epochs = 50
    batch_size = 32
    batch_size_trees = 128

    if features is not None:
        raw_data = raw_data if not is_multiclass else raw_data[0]
        features_vals = features.value
        frame_cols = raw_data.columns.tolist()
        criteria_list = [x for x in frame_cols if x in features_vals]
        raw_data = raw_data[criteria_list]
    train_pd_ds = raw_data.head(ds_len - pct_len).copy(deep=True)
    features: list = train_pd_ds.columns.tolist()
    features.remove('estimated')
    test_pd_ds = raw_data.tail(pct_len).copy(deep=True)
    tf_train = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(train_pd_ds[features].values, tf.float16),
                tf.cast(train_pd_ds['estimated'].values, tf.uint8)
            )
        )
    )
    tf_test = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(test_pd_ds[features].values, tf.float16),
                tf.cast(test_pd_ds['estimated'].values, tf.uint8)
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
            tf_train_nn_single = tf_train.shuffle(
                num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
            tf_test_nn = tf_test.shuffle(
                num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
            tf_val_nn=tf_train_nn_single.take(8)
            input_layer=ks.layers.Input((len(features),))
            input_flat= ks.layers.Flatten(input_shape=(len(features),))(input_layer)
            hidden_norm=ks.layers.BatchNormalization(trainable=False)(input_flat)
            hidden_3_1 = ks.layers.Dense(1024, activation='relu')(hidden_norm)
            hidden_4_1 = ks.layers.Dense(512, activation='relu')(hidden_3_1)
            hidden_5_1 = ks.layers.Dense(128, activation='relu')(hidden_4_1)
            hidden_6_1 = ks.layers.Dense(64, activation='relu')(hidden_5_1)
            #out_layer=Concatenate()([hidden_3_1,hidden_4_1,hidden_5_1,hidden_6_1])
            out_layer_total=ks.layers.Dense(1,activation='sigmoid')(hidden_6_1)
            model_class=Model(input_layer,out_layer_total)
            model_class.compile(optimizer="adam",loss='binary_crossentropy',metrics=['binary_accuracy'])
            model_class.summary()
            history = model_class.fit(tf_train_nn_single, epochs=num_epochs, validation_data=tf_val_nn,
                                      steps_per_epoch=math.ceil(num_train_examples / batch_size),
                                      shuffle=True)
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
        tf_train_nn = tf_train.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        tf_test_nn = tf_test.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        input_nodes = int(len(features))
        output_nodes = len(inp_ds[2])
        tf_val_nn = tf_train_nn.take(8)
        try:
            model_multiclass = ks.Sequential([
                ks.layers.Input((len(features),)),
                ks.layers.Flatten(input_shape=(input_nodes,)),
                ks.layers.BatchNormalization(),
                ks.layers.Dense(2048, activation='relu', kernel_initializer="he_normal"),
                ks.layers.Dropout(0.3),
                ks.layers.Dense(512, activation='softmax', kernel_initializer="he_normal"),
                ks.layers.Dropout(0.3),
                ks.layers.Dense(15, activation='relu'),
                ks.layers.Dense(output_nodes - 2, activation='relu', kernel_initializer="uniform"),
                ks.layers.Dense(output_nodes, activation='sigmoid'),
            ])
            model_multiclass.compile("adam", loss="mse", metrics=['mae'])
            model_multiclass.summary()
            history = model_multiclass.fit(tf_train_nn, batch_size=batch_size, epochs=num_epochs,
                                           validation_data=tf_val_nn,
                                           steps_per_epoch=math.ceil(num_train_examples / batch_size)
                                           ,shuffle=True)
            evaluate_multiclass = model_multiclass.evaluate(tf_test_nn)
            evaluations['regression'] = {
                'loss': evaluate_multiclass[0],
                'mse': evaluate_multiclass[1]
            }

            do_plot_history_seq(history, "Sequential NN muliclass classification", metric='mae')
        except Exception as ex:
            print(ex.__str__())
