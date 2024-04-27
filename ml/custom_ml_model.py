from keras import Model
import tensorflow as tf
from tensorflow import keras as ks


def create_baseline_model(hidden_units, features, output):
    inputs = ks.layers.Input((len(features),))
    features = ks.layers.BatchNormalization()(inputs)
    idx = 1
    for units in hidden_units:
        features = ks.layers.Dense(units, activation="leaky_relu")(features)
        idx += 1
        if idx % 2 == 0:
            features = ks.layers.Dropout(0.2)(features)
    # The output is deterministic: a single point estimate
    features = tf.keras.layers.ActivityRegularization(l1=0.001, l2=0.001)(features)
    outputs = ks.layers.Dense(units=output, activation="softmax")(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model
