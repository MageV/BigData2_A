"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import math
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import os
from ml.ml_utils import *
from providers.df import df_remove_outliers
from providers.ui import *
from config.appconfig import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001 * 10 ** (epoch / 20)


def tf_learn_model(inp_ds, pct_val,pct_train, is_multiclass, features=None, skiptree=False, tface=None):
    raw_data = inp_ds.copy(deep=True) if not is_multiclass else inp_ds[0].copy(deep=True)
    if tface is not None:
        raw_data = raw_data.loc[raw_data['typeface'] == tface.value]
        raw_data.drop('typeface', inplace=True, axis=1)

    evaluations = dict()
    ds_len = len(raw_data)
    pct_val_len = int(round(pct_val * ds_len, 0))
    pct_test_len=int(round(pct_train * ds_len, 0))
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
    train_len= ds_len - pct_val_len-pct_test_len
    train_pd_ds = raw_data.head(train_len).copy(deep=True)
    features: list = train_pd_ds.columns.tolist()
    features.remove('estimated')
    val_pd_ds = raw_data[train_len:train_len+pct_val_len].copy(deep=True)
    test_pd_ds=raw_data[train_len+pct_val_len:train_len+pct_val_len+pct_test_len]
    tf_train = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(train_pd_ds[features].values, tf.float32),
                tf.cast(train_pd_ds['estimated'].values, tf.uint8)
            )
        )
    )
    tf_val = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(val_pd_ds[features].values, tf.float32),
                tf.cast(val_pd_ds['estimated'].values, tf.uint8)
            )
        )
    )
    tf_test=(
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(test_pd_ds[features].values, tf.float32),
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
                confmx_hboost = tf.math.confusion_matrix(predict_hboost, val_pd_ds['estimated'].values)
                print(confmx_hboost)
                do_plot_train_trees(model_rf, "RandomForestClassifier binary classification")
                do_plot_train_trees(model_hboost, "GradientBoostTrees binary classification")
            # do_plot_conf_mx(df_mx_hboost,"Confusion matrix GradientBoostTrees binary classification")
            except Exception as ex:
                print(ex.__str__())
        try:
            hidden_units = [1024, 512, 128, 64, 4]
            tf_train_nn_single = tf_train.shuffle(
                num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
            tf_val_nn = tf_val.shuffle(
                num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
            tf_test_nn = tf_test.shuffle(
                num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
            model_class = create_baseline_model(hidden_units, features, 1)
            model_class.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])
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
        hidden_units = [1024, 512, 128, 64, 12]
        batch_size = 1
        tf_train_nn = tf_train.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        tf_val_nn = tf_val.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        input_nodes = int(len(features))
        output_nodes = len(inp_ds[2])
        tf_test_nn = tf_test.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        try:
            model_multiclass = create_baseline_model(features=features, hidden_units=hidden_units, output=output_nodes)
            model_multiclass.compile("adam", loss='mse', metrics=['mae'])
            model_multiclass.summary()
            history = model_multiclass.fit(tf_train_nn, batch_size=batch_size, epochs=num_epochs,
                                           validation_data=tf_val_nn,
                                           steps_per_epoch=math.ceil(num_train_examples / batch_size)
                                           , shuffle=True)
            evaluate_multiclass = model_multiclass.evaluate(tf_test_nn)
            evaluations['regression'] = {
                'loss': evaluate_multiclass[0],
                'mse': evaluate_multiclass[1]
            }

            do_plot_history_seq(history, "Sequential NN muliclass classification", metric='mae')
        except Exception as ex:
            print(ex.__str__())
