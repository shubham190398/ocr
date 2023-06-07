import os
from tqdm import tqdm
import tensorflow as tf

# noinspection PyBroadException
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from dataloader import DataLoader
from preprocessors import ImageReader
from image_formats import CVImage
from transformers import ImageResizer, LabelIndexer, LabelPadding
from model_utils import CTCloss, Model2Onnx, TrainLogger, ErrorMetric
from model import train_model
