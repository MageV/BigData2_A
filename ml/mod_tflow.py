"""
history_size — это размер последнего временного интервала,
target_size – аргумент, определяющий насколько далеко в будущее модель должна научиться прогнозировать
"""
import asyncio
import math
import tensorflow_decision_forests as tfdf
from tensorflow import keras as ks
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from apputils.log import write_log
from ml.tf_ml_models import *
from providers.ui import *
from config.appconfig import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001 * 10 ** (epoch / 20)


def tf_learn_model(inp_ds, pct_val, pct_train, classifier, tface=None):
    raw_data = inp_ds.copy(deep=True) if classifier in [TF_OPTIONS.TF_TREES_BINARY, TF_OPTIONS.TF_NN_BINARY,
                                                        TF_OPTIONS.TF_LSTM] \
        else inp_ds[0].copy(deep=True)
    if tface is not None:
        raw_data = raw_data.loc[raw_data['typeface'] == tface.value]
        raw_data.drop('typeface', inplace=True, axis=1)
    raw_data['dead_credits'] = raw_data['debt_mass'] / raw_data['credits_mass']
    raw_data = raw_data.drop(['debt_mass', 'credits_mass'], axis=1)
    features: list = raw_data.columns.tolist()
    features.remove('estimated')
    num_train_examples = 6000
    num_epochs = 30
    batch_size = 32
    batch_size_trees = 128
    tf_train, tf_val, tf_test, test_pd_ds, val_pd_ds = _prepare_as_tensors(raw_data, pct_train, pct_val, features)
    if classifier == TF_OPTIONS.TF_TREES_BINARY:
        run_tf_trees_binary(batch_size_trees, test_pd_ds, val_pd_ds, tf_train, tf_test, features)
    elif classifier == TF_OPTIONS.TF_NN_BINARY:
        run_tf_nn_binary(tf_train, tf_val, tf_test, num_train_examples, batch_size, features, num_epochs)
    elif classifier == TF_OPTIONS.TF_NN_MULTU:
        run_tf_nn_multi(raw_data, features, num_epochs, inp_ds[2] )
    elif classifier == TF_OPTIONS.TF_LSTM:
        run_tf_lstm(raw_data, features, 1, batch_size, num_epochs)


def _prepare_as_tensors(frame, pct_train, pct_val, features):
    frame = frame.sample(frac=1)
    ds_len = len(frame)
    pct_val_len = int(round(pct_val * ds_len, 0))
    pct_test_len = int(round(pct_train * ds_len, 0))
    train_len = ds_len - pct_val_len - pct_test_len
    train_pd_ds = frame.head(train_len).copy(deep=True)
    val_pd_ds = frame[train_len:train_len + pct_val_len].copy(deep=True)
    test_pd_ds = frame[train_len + pct_val_len:train_len + pct_val_len + pct_test_len]
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
    tf_test = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(test_pd_ds[features].values, tf.float32),
                tf.cast(test_pd_ds['estimated'].values, tf.uint8)
            )
        )
    )
    tf_train = tf_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    tf_val = tf_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    tf_test = tf_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return tf_train, tf_val, tf_test, test_pd_ds, val_pd_ds


def run_tf_trees_binary(batch_size_trees, test_pd, val_pd, tf_train, tf_test, features):
    try:
        tf_train_trees = tf_train.batch(batch_size_trees)
        tf_test_trees = tf_test.batch(batch_size_trees)
        tuner = tfdf.tuner.RandomSearch(num_trials=200,
                                        trial_num_threads=12)
        model_rf = tfdf.keras.RandomForestModel(tuner=tuner, verbose=2)
        model_hboost = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, verbose=2)
        model_rf.fit(tf_train_trees)
        model_hboost.fit(tf_train_trees)
        model_hboost.compile(metrics=['accuracy'])
        model_rf.compile(metrics=["accuracy"])
        evaluate_rf = model_rf.evaluate(tf_test_trees, return_dict=True)
        evaluate_hboost = model_hboost.evaluate(tf_test_trees, return_dict=True)
        predict_rf = model_rf.predict(test_pd[features].values)
        predict_hboost = model_hboost.predict(test_pd[features].values)
        do_plot_train_trees(model_rf, "RandomForestClassifier binary classification")
        do_plot_train_trees(model_hboost, "GradientBoostTrees binary classification")
        # do_plot_conf_mx(df_mx_hboost,"Confusion matrix GradientBoostTrees binary classification")
        return model_rf, model_hboost
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex.__str__()}', severity=SEVERITY.ERROR))
        return None


def run_tf_nn_binary(tf_train, tf_val, tf_test, num_train_examples, batch_size, features, num_epochs):
    lr_schedule = ks.optimizers.schedules.PolynomialDecay(
        1e-3, 10000, 1e-5, 0.5, True)
    try:
        hidden_units = [1024, 511, 128, 63, 4]
        tf_train_nn_single = tf_train.shuffle(
            num_train_examples, reshuffle_each_iteration=True).repeat().batch(batch_size)
        tf_val_nn = tf_val.shuffle(
            num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
        tf_test_nn = tf_test.shuffle(
            num_train_examples, reshuffle_each_iteration=True).batch(batch_size)
        model_class = create_baseline_model_binary(hidden_units, features, 1)
        model_class.compile(optimizer="adam",
                            loss=ks.losses.BinaryCrossentropy(),
                            metrics=['auc', 'FalsePositives',
                                     'TruePositives', 'FalseNegatives', 'TrueNegatives'])
        model_class.summary()
        history = model_class.fit(tf_train_nn_single, epochs=num_epochs, validation_data=tf_val_nn,
                                  steps_per_epoch=math.ceil(num_train_examples / batch_size),
                                  shuffle=True)
        estim = model_class.evaluate(tf_test_nn, batch_size=batch_size, return_dict=True)
        asyncio.run(write_log(message=f"    0      1\n"
                                      f"0   {estim['TrueNegatives']}     {estim['FalseNegatives']}\n"
                                      f"1   {estim['FalsePositives']}     {estim['TruePositives']}",
                              severity=SEVERITY.INFO))
        do_plot_history_seq(history, "Sequential NN binary classification", ["auc"])
        return model_class
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex.__str__()}', severity=SEVERITY.ERROR))
        return None


def run_tf_nn_multi(ds, features, num_epochs, outputs_len, batch_size=1):
    hidden_units = [len(outputs_len) * 4, len(outputs_len) * 2]
    output_nodes = outputs_len
    X_train, X_test, y_train, y_test = train_test_split(ds[features].values, ds['estimated'].values,
                                                        test_size=0.2, random_state=42, shuffle=True)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    try:
        model_multiclass = create_baseline_model_multi(features=features, hidden_units=hidden_units,
                                                       output=len(output_nodes))
        model_multiclass.compile(optimizer="adam",
                                 loss="mae", metrics=['mse'])
        model_multiclass.summary()
        history = model_multiclass.fit(X_train,y_train, epochs=num_epochs,
                                       validation_split=0.25, shuffle=True)
        estim = model_multiclass.evaluate(X_test,y_test, batch_size=batch_size,
                                          return_dict=True)
        do_plot_history_seq(history, "Sequential NN muliclass classification", metric=['mse'])
        return model_multiclass
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex.__str__()}', severity=SEVERITY.ERROR))
        return None


def run_tf_lstm(ds, features, output, batch_size, numepochs):
    X_train, X_test, y_train, y_test = train_test_split(ds[features].values, ds['estimated'].values,
                                                        test_size=0.15, random_state=42, shuffle=True)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model = create_lstm_model(features, output, ds)
    model.compile(loss='binary_crossentropy', optimizer="lion",
                  metrics=['auc', 'FalsePositives',
                           'TruePositives', 'FalseNegatives', 'TrueNegatives'])
    model.summary()
    history = model.fit(X_train, y_train, validation_split=0.25, epochs=numepochs, batch_size=batch_size)
    estim = model.evaluate(X_test, y_test, batch_size=batch_size, return_dict=True)
    asyncio.run(write_log(message=f"    0        1\n"
                                  f"0   {estim['TrueNegatives']}     {estim['FalseNegatives']}\n"
                                  f"1   {estim['FalsePositives']}     {estim['TruePositives']}",
                          severity=SEVERITY.INFO))
    do_plot_history_seq(history, "Sequential NN binary classification", ["auc"])
    pass
