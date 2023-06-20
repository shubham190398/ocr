import keras.backend
from keras import layers
from keras.models import Model


def inference_model(input_dim, output_dim, kernel_size=(3, 3), activation="relu", padding="same", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    conv1 = layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
    pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding)(pool1)

    conv3 = layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding)(conv2)
    pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding)(pool3)

    conv5 = layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding)(conv4)

    conv6 = layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding)(conv5)
    pool6 = layers.MaxPool2D(pool_size=(2, 1))(conv6)

    conv7 = layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding)(pool6)
    bn7 = layers.BatchNormalization()(conv7)

    conv8 = layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding)(bn7)
    bn8 = layers.BatchNormalization()(conv8)

    conv9 = layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding)(bn8)
    bn9 = layers.BatchNormalization()(conv9)
    pool9 = layers.MaxPool2D(pool_size=(2, 1))(bn9)

    conv10 = layers.Conv2D(512, kernel_size=(2, 2), activation='relu')(pool9)

    squeezed = layers.Lambda(lambda x: keras.backend.squeeze(x, 1))(conv10)

    blstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm1d = layers.Dropout(dropout)(blstm1)

    blstm2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm1d)
    blstm2d = layers.Dropout(dropout)(blstm2)

    blstm3 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm2d)
    blstm3d = layers.Dropout(dropout)(blstm3)

    blstm4 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm3d)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm4)

    text_model = Model(inputs=inputs, outputs=output)

    return text_model


def ctc(args):
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
