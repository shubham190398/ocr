import ast
import numpy as np
import cv2
from text_extraction_utils import encode_labels, preprocess_img, create_vocab
from text_extraction_model import training_model
import os
import random
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint


def train(image_dir, annotation_dir):
    image_names = os.listdir(image_dir)
    image_names = [file.split(".")[0] for file in image_names]
    images = []
    annotations = []
    label_length = []
    input_length = []

    max_label_length = 0

    vocab = create_vocab(annotation_dir)
    print("Vocab created")

    random.shuffle(image_names)

    for filename in image_names:
        original_img = cv2.imread(f"{image_dir}/{filename}.png", cv2.IMREAD_GRAYSCALE)

        with open(f"{annotation_dir}/{filename}.txt", "r") as f:
            word_dict = ast.literal_eval(f.readlines()[0])

            for index, coordinate in enumerate(word_dict['bboxes']):
                x1, y1, x2, y2 = coordinate
                img = original_img[y1:y2, x1:x2]
                img = preprocess_img(img, (128, 32))
                img = np.expand_dims(img, axis=-1)
                img = img/255

                annotation = word_dict['words'][index].lower()

                if len(annotation) > max_label_length:
                    max_label_length = len(annotation)

                images.append(img)
                label_length.append(len(annotation))
                input_length.append(31)
                annotations.append(encode_labels(annotation, vocab))

    dataset_length = len(images)
    print(f"Length of dataset is {dataset_length}")

    split_length = int(0.95*dataset_length)
    train_images, valid_images = images[:split_length], images[split_length:]
    train_input_length, valid_input_length = input_length[:split_length], input_length[split_length:]
    train_label_length, valid_label_length = label_length[:split_length], label_length[split_length:]
    train_annotations, valid_annotations = annotations[:split_length], annotations[split_length:]

    train_padded_annots = pad_sequences(train_annotations, maxlen=max_label_length, padding='post', value=len(vocab))
    valid_padded_annots = pad_sequences(valid_annotations, maxlen=max_label_length, padding='post', value=len(vocab))

    model = training_model(input_dim=(32, 128, 1), output_dim=len(vocab), max_len=max_label_length)
    output_path = "/content/text_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=output_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    train_images, valid_images = np.array(train_images), np.array(valid_images)
    train_input_length, valid_input_length = np.array(train_input_length), np.array(valid_input_length)
    train_label_length, valid_label_length = np.array(train_label_length), np.array(valid_label_length)

    batch_size = 1024
    epochs = 100

    model.fit(
        x=[train_images, train_padded_annots, train_input_length, train_label_length],
        y=np.zeros(len(train_images)),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([valid_images, valid_padded_annots, valid_input_length, valid_label_length],
                         [np.zeros(len(valid_images))]),
        verbose=1,
        callbacks=callbacks_list
    )


def main():
    image_dir = ""
    annotation_dir = ""
    train(image_dir, annotation_dir)


if __name__ == "__main__":
    main()
