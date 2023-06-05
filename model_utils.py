from keras import layers


def activation_layer(layer, activation="relu", alpha=0.1):
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU()(layer)
    else:
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer


def residual_block(x, filter_num, strides=2, kernel_size=3, skip_conv=True, padding="same",
                   kernel_initializer="he_uniform", activation="leaky_relu", dropout=0.2):
    skip_x = x

    x = layers.Conv2D(filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    x = layers.Conv2D(filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer)
    x = layers.BatchNormalization()(x)

    if skip_conv:
        skip_x = layers.Conv2D(filter_num, 1, padding=padding, strides=strides,
                               kernel_initializer=kernel_initializer)(skip_x)

    x = layers.add()([x, skip_x])
    x = activation_layer(x, activation=activation)

    if dropout:
        x = layers.Dropout(dropout)(x)

    return x
