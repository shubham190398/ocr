import os
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from dataloader import DataLoader
from preprocessors import ImageReader
from image_formats import CVImage
from transformers import ImageResizer, LabelIndexer, LabelPadding
from model_utils import CTCloss, Model2Onnx, TrainLogger, ErrorMetric
from model import train_model
from configs import ModelConfigs
import stow
import tarfile
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


# noinspection PyBroadException
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass


def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    zipfile = ZipFile(BytesIO(data))
    zipfile.extractall(path=extract_to)
