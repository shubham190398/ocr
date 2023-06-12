from keras import layers
import copy


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

    y = copy.deepcopy(x)

    if dropout:
        drop = layers.Dropout(dropout=0.5)(y)

        if pooling:
            y = layers.MaxPooling2D(pool_size=(2, 2))(drop)

            return y, drop

    if pooling:
        y = layers.MaxPooling2D(pool_size=(2, 2))(y)

    return y, x
