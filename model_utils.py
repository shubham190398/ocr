import tensorflow as tf
from keras import layers
from keras.metrics import Metric


def activation_layer(layer, activation="relu", alpha=0.1):
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer


def residual_block(x, filter_num, strides=2, kernel_size=3, skip_conv=True, padding="same",
                   kernel_initializer="he_uniform", activation="leaky_relu", dropout=0.2):
    skip_x = x

    x = layers.Conv2D(filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    x = layers.Conv2D(filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)

    if skip_conv:
        skip_x = layers.Conv2D(filter_num, 1, padding=padding, strides=strides,
                               kernel_initializer=kernel_initializer)(skip_x)

    x = layers.add()([x, skip_x])
    x = activation_layer(x, activation=activation)

    if dropout:
        x = layers.Dropout(dropout)(x)

    return x


# noinspection PyAbstractClass
class CTCloss(tf.keras.losses.Loss):
    def __init__(self, name: str = "CTCLoss") -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss


class ErrorMetric(Metric):
    def __init__(self, padding_token, name="ErrorMetric", **kwargs):
        super(ErrorMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.keras.backend.shape(y_pred)

