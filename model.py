from keras import layers
from keras.models import Model
from model_utils import residual_block


def train_model(input_dim, output_dim, activation="leaky_relu")