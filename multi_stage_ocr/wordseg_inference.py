import cv2
import numpy as np
import matplotlib.pyplot as plt
from word_segmentor_model import unet


line_seg_model = unet(pretrained_weights="models/50.h5")
word_seg_model = unet(pretrained_weights="models/wordseg-35.h5")


def line_detection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = line_seg_model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    img = cv2.normalize(pred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)

    original_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    (h, w) = original_img.shape[:2]
    re_h, re_w = 512, 512
    factor_h, factor_w = h / float(re_h), w / float(re_w)

    coordinates = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        coordinates.append((int(x * factor_w), int(y * factor_h), int((x + w) * factor_w), int((y + h) * factor_h)))

    count = 1

    for i in range(len(coordinates)):
        coord = coordinates[i]
        line_img = original_img[coord[1]:coord[3], coord[0]:coord[2]].copy()
        cv2.imwrite(f"results/line_images/{count}.jpg", line_img)
