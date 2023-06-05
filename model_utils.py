import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model


def activation_layer(layer, activation="relu", alpha=0.1):
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU()(layer)
    else:
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer
