import os
import ast
import cv2
import numpy as np


def preprocess_img(img, imgSize=(128, 32)):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    w_target, h_target = imgSize
    h, w = img.shape
    fx, fy = w / float(w_target), h / float(h_target)
    f = max(fx, fy)

    target_size = (max(min(w_target, int(w / f)), 1), max(min(h_target, int(h / f)), 1))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img


def create_vocab(path):
    textfiles = os.listdir(path)
    vocab = set()

    for file in textfiles:

        with open(f"{path}/{file}", "r") as f:
            print(f"Reading file {file}")
            name = file.split(".")[0]
            img = cv2.imread(f"C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD\\{name}.png", 0)
            word_dict = ast.literal_eval(f.readlines()[0])
            for index, coordinates in enumerate(word_dict['bboxes']):
                x1, y1, x2, y2 = coordinates
                im = img[y1:y2, x1:x2]
                im = preprocess_img(im)
                cv2.imshow("im", im)
                print(im.shape)
                print(word_dict['words'][index])
                cv2.waitKey(0)

        break


create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")
