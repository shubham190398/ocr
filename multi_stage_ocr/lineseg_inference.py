from line_segmentor_model import unet
import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect():
    model = unet(pretrained_weights="models/50.h5")
    img = cv2.imread("dataset/invoices/335.jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    img = cv2.resize(img, (512, 512))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)
    # cv2.imshow("pred", pred)
    # cv2.waitKey(0)
    plt.imsave("results/335_mask.JPG", pred)


def main():
    detect()
    img = cv2.imread("results/335_mask.JPG", cv2.IMREAD_GRAYSCALE)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    original_img = cv2.imread("dataset/invoices/335.jpg")
    original_img = cv2.resize(original_img, (512, 512))

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), 255, 1)

    cv2.imwrite("results/335_contours.JPG", original_img)


if __name__ == "__main__":
    main()
