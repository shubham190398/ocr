from model_utils import conv_block
from keras.layers import Conv2D, Input
from keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)

    x, conv1 = conv_block(inputs, filter_num=64, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    x, conv2 = conv_block(x, filter_num=128, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    x, conv3 = conv_block(x, filter_num=256, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    x, drop4 = conv_block(x, filter_num=512, kernel_stride=3, dropout=True, pooling=True, upsampling=False)

    x, drop5 = conv_block(x, filter_num=1024, kernel_stride=3, dropout=True, pooling=False, upsampling=False)

    x, conv6 = conv_block(x, filter_num=512, kernel_stride=3, pooling=False, upsampling=True, merge=drop4)

    x, conv7 = conv_block(x, filter_num=256, kernel_stride=3, pooling=False, upsampling=True, merge=conv3)

    x, conv8 = conv_block(x, filter_num=128, kernel_stride=3, pooling=False, upsampling=True, merge=conv2)

    x, conv9 = conv_block(x, filter_num=64, kernel_stride=3, pooling=False, upsampling=True, merge=conv1)
