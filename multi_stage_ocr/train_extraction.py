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






def main():
    image_dir = ""
    annotation_dir = ""
    train(image_dir, annotation_dir)


if __name__ == "__main__":
    main()
