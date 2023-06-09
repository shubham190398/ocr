from keras import layers
from keras.models import Model
from model_utils import residual_block, activation_layer


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):

    # CNN
    inputs = layers.Input(shape=input_dim, name="input")
    input = layers.Lambda(lambda x: x/255)(inputs)

    x1 = residual_block(input, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x8 = residual_block(x7, 256, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x9 = residual_block(x8, 256, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Reshape
    squeezed_layers = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    # RNN
    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed_layers)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm)

    # Dense
    d = layers.Dense(256)(blstm)
    d = activation_layer(d, activation="leaky_relu")
    d = layers.Dropout(dropout)(d)

    # Classification
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(d)

    model = Model(inputs=inputs, outputs=output)

    return model
