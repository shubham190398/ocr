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



def main():
    image_dir = ""
    annotation_dir = ""
    train(image_dir, annotation_dir)


if __name__ == "__main__":
    main()
