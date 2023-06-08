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


# noinspection PyBroadException
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass


config = ModelConfigs()
data_path = ""
val_annotation_path = data_path + "/annotation_val.txt"
train_annotation_path = data_path + "/annotation_train.txt"


def read_annotation_file(annotation_path):
    dataset, vocab, max_len = [], set(), 0
    with open(annotation_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            image_path = data_path + line[0][1:]
            label = line[0].split("_")[1]
            dataset.append([image_path, label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
    return dataset, sorted(vocab), max_len


train_dataset, train_vocab, max_train_len = read_annotation_file(train_annotation_path)
val_dataset, val_vocab, max_val_len = read_annotation_file(val_annotation_path)

config.vocab = "".join(train_vocab)
config.max_text_length = max(max_train_len, max_val_len)
config.save()


train_data_loader = DataLoader(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=config.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(config.width, config.height),
        LabelIndexer(config.vocab),
        LabelPadding(max_word_length=config.max_text_length, padding_value=len(config.vocab))
    ]
)

val_data_loader = DataLoader(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=config.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(config.width, config.height),
        LabelIndexer(config.vocab),
        LabelPadding(max_word_length=config.max_text_length, padding_value=len(config.vocab))
    ]
)

model = train_model(
    input_dim=(config.height, config.width, 3),
    output_dim=len(config.vocab)
)
