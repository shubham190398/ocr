from keras import layers
import copy


def conv_block(
        x,
        filter_num,
        kernel_stride,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
):

    x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                      padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(filter_num, kernel_stride, activation=activation,
                      padding=padding, kernel_initializer=kernel_initializer)(x)

    return x
