from line_segmentor_model import unet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def detect(path, img_name):
    model = unet(pretrained_weights="models/text_seg_model.h5")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
    plt.imsave(f"results/{img_name}_mask.JPG", pred)


def main(path, img_name):
    detect(path, img_name)
    img = cv2.imread(f"results/{img_name}_mask.JPG", cv2.IMREAD_GRAYSCALE)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    original_img = cv2.imread(path)
    original_img = cv2.resize(original_img, (512, 512))

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), 255, 1)

    cv2.imwrite(f"results/{img_name}_contours.JPG", original_img)


if __name__ == "__main__":
    dir = os.listdir("dataset/clientdata")
    for file in dir:
        main("dataset/clientdata/" + file, file.split(".")[0])
