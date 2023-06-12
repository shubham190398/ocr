from model_utils import conv_block
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.models import Model


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)

    conv1 = conv_block(inputs, filter_num=64, kernel_stride=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, filter_num=128, kernel_stride=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, filter_num=256, kernel_stride=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, filter_num=512, kernel_stride=3)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_block(pool4, filter_num=1024, kernel_stride=3)
    drop5 = Dropout(0.4)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D()(drop5))
    merge1 = concatenate([drop4, up6], axis=3)
    conv6 = conv_block(merge1, filter_num=512, kernel_stride=3)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D()(conv6))
    merge2 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge2, filter_num=256, kernel_stride=3)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D()(conv7))
    merge3 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge3, filter_num=128, kernel_stride=3)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D()(conv8))
    merge4 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge4, filter_num=64, kernel_stride=3)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


model_unet = unet()
model_unet.summary()
