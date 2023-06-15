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


def nearest_10(num):
    return int(np.ceil(num / 10.0)) * 10


def pad_image(img):
    h, w = img.shape[0], img.shape[1]

    if h < 512:
        pad = np.ones((512 - h, w)) * 255
        img = np.concatenate((img, pad))
        h = 512
    else:
        pad = np.ones((nearest_10(h) - h, w)) * 255
        img = np.concatenate((img, pad))
        h = nearest_10(h)

    if w < 512:
        pad = np.ones((h, 512 - w)) * 255
        img = np.concatenate((img, pad), axis=1)
    else:
        pad = np.ones((h, nearest_10(w) - w)) * 255
        img = np.concatenate((img, pad), axis=1)

    return img


def pad_seg(seg):
    h, w = seg.shape[0], seg.shape[1]

    if h < 512:
        pad = np.zeros((512 - h, w))
        seg = np.concatenate((seg, pad))
        h = 512
    else:
        pad = np.zeros((nearest_10(h) - h, w))
        seg = np.concatenate((seg, pad))
        h = nearest_10(h)

    if w < 512:
        pad = np.zeros((h, 512 - w))
        seg = np.concatenate((seg, pad), axis=1)
    else:
        pad = np.zeros((h, nearest_10(w) - w))
        seg = np.concatenate((seg, pad), axis=1)

    return seg


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
            mask = mask / 255

            images.append(im)
            masks.append(mask)

        yield np.array(images), np.array(masks)
