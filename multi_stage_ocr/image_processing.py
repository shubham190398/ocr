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


def preprocessing(dir_path):
    image_list = os.listdir(dir_path)
    for image_path in image_list:
        im = cv2.imread(f"{dir_path}/{image_path}")
        im[im > 0] = 255
        cv2.imwrite(f"{dir_path}/{image_path}", im)
        print(f"Writing image {image_path}")


preprocessing("C:\\Users\\Kare4U\\Downloads\\PageSegExtended\\customdataset_extending_pageseg\\PageSeg")
