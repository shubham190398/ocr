import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random


def display(img, seg_img):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.show()


def preprocessing(dir_path):
    image_list = os.listdir(dir_path)
    for image_path in image_list:
        im = cv2.imread(f"{dir_path}/{image_path}")
        im[im > 0] = 255
        cv2.imwrite(f"{dir_path}/{image_path}", im)
        print(f"Writing image {image_path}")


def batch_generator(dir_path, image_names, batch_size):
    while True:
        images = []
        masks = []

        for i in range(batch_size):
            im_name = random.choice(image_names)
            im = cv2.imread(f"{dir_path}/{im_name}.jpg", cv2.IMREAD_GRAYSCALE)
            _, im = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV)
            im = cv2.resize(im, (512, 512))
            im = np.expand_dims(im, axis=-1)
            im = im / 255

            mask = cv2.imread(f"{dir_path}/{im_name}_mask.png", cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512))
            mask /= 255

            images.append(im)
            masks.append(mask)

        yield np.array(images), np.array(masks)
