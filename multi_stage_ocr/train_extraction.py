import ast

import numpy as np
import cv2
from text_extraction_utils import encode_labels, preprocess_img, create_vocab
import os
import random


def train(image_dir, annotation_dir):
    image_names = os.listdir(image_dir)
    image_names = [file.split(".")[0] for file in image_names]
    images = []
    annotations = []
    label_length = []
    original_text = []
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
                original_text.append(annotation)
                label_length.append(len(annotation))
                input_length.append(len(input_length))
                annotations.append(encode_labels(annotation, vocab))

    dataset_length = len(images)
    print(f"Length of dataset is {dataset_length}")

    split_length = int(0.95*dataset_length)
    train_images, valid_images = images[:split_length], images[split_length:]
    train_text, valid_text = original_text[:split_length], images[split_length:]
    train_input_length, valid_input_length = input_length[:split_length], input_length[split_length:]
    train_label_length, valid_label_length = label_length[:split_length], label_length[split_length:]
    train_annotations, valid_annotations = annotations[:split_length], annotations[split_length:]




def main():
    image_dir = ""
    annotation_dir = ""
    train(image_dir, annotation_dir)


if __name__ == "__main__":
    main()
