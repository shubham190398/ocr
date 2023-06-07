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
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse,
                                                   tf.not_equal(predicted_labels_sparse.values, -1))
        true_labels_sparse = tf.sparse.retain(true_labels_sparse,
                                              tf.not_equal(true_labels_sparse.values, self.padding_token))

        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))

        self.batch_counter.assign_add(len(y_true))

        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32)))


