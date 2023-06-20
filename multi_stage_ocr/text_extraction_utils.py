import os
import ast
import cv2
import numpy as np


def create_vocab(path):
    textfiles = os.listdir(path)
    vocab = set()

    for file in textfiles:

        with open(f"{path}/{file}", "r") as f:
            print(f"Reading file {file}")
            word_dict = ast.literal_eval(f.readlines()[0])

            for word in word_dict['words']:
                vocab.update(word)

    return sorted(vocab)


def preprocess_img(img, imgSize=(128, 32)):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    w_target, h_target = imgSize
    h, w = img.shape
    fx, fy = w/float(w_target), h/float(h_target)
    f = max(fx, fy)

    target_size = (max(min(w_target, int(w/f)), 1), max(min(h_target, int(h/f)), 1))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    target = np.ones([h_target, w_target]) * 255
    target[0:target_size[1], 0:target_size[0]] = img

    return target


def encode_labels(text, vocab):
    digit_list = []

    for index, char in enumerate(text):
        try:
            digit_list.append(vocab.index(char))
        except:
            print(f"{char} is not in vocab")

    return digit_list


# print("".join(create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")))
