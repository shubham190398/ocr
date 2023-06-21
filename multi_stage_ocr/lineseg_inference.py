from line_segmentor_model import unet
import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect():
    model = unet(pretrained_weights="models/50.h5")
    img = cv2.imread("dataset/LineSeg/356.JPG", cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)
    plt.imsave("results/356_mask.JPG", pred)


def main():
    detect()
    img = cv2.imread("results/356_mask.JPG", cv2.IMREAD_GRAYSCALE)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    original_img = cv2.imread("dataset/LineSeg/356.JPG")
    original_img = cv2.resize(original_img, (512, 512))

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), 255, 1)

    cv2.imwrite("results/356_contours.JPG", original_img)


if __name__ == "__main__":
    main()
