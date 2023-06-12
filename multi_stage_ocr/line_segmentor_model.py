from model_utils import conv_block
from keras.layers import Conv2D, Input
from keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)

    conv1 = conv_block(inputs, filter_num=64, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    conv2 = conv_block(conv1, filter_num=128, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    conv3 = conv_block(conv2, filter_num=256, kernel_stride=3, dropout=False, pooling=True, upsampling=False)

    x = conv_block(x, filter_num=512, kernel_stride=3, dropout=True, pooling=True, upsampling=False)

    x = conv_block(x, filter_num=1024, kernel_stride=3, dropout=True, pooling=False, upsampling=False)

    x = conv_block(x, filter_num=512, kernel_stride=3, dropout=False, pooling=False, upsampling=True)

    x = conv_block(x, filter_num=256, kernel_stride=3, dropout=False, pooling=False, upsampling=True)

    x = conv_block(x, filter_num=128, kernel_stride=3, dropout=False, pooling=False, upsampling=True)

    x = conv_block(x, filter_num=64, kernel_stride=3, dropout=False, pooling=False, upsampling=True)