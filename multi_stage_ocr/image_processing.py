import matplotlib.pyplot as plt
import os
import cv2


def display(img, seg_img):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.show()



