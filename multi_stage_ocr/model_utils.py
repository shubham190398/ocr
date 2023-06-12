from keras import layers


def conv_block(
        x,
        filter_num,
        kernel_stride,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        dropout=False,
        pooling=True,
        upsampling=False,
        merge=None
):
    if not upsampling:
        x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                          padding=padding, kernel_initializer=kernel_initializer)(x)
        x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                          padding=padding, kernel_initializer=kernel_initializer)(x)
    else:
        up = layers.Conv2D(filter_num, kernel_stride-1, activation=activation, padding=padding,
                           kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2, 2)))(x)
        x = layers.concatenate([merge, up], axis=3)
        x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                          padding=padding, kernel_initializer=kernel_initializer)(x)
        x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                          padding=padding, kernel_initializer=kernel_initializer)(x)

    if dropout:
        x = layers.Dropout(dropout=0.5)(x)

    if pooling:
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    return x
