import cv2
from image_processing import pad_image
import numpy as np


image_path = "9"
original_img = cv2.imread(f"results/line_images/{image_path}.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread(f"results/word_segs/{image_path}_mask.jpg", cv2.IMREAD_GRAYSCALE)
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
original_img = pad_image(original_img)
(h, w) = original_img.shape[:2]
factor_h, factor_w = h/512.0, w/512.0
original_img_copy = np.stack((original_img,)*3, axis=-1)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    points = []
    for point in c:
        x = int(point[0][0]*factor_w)
        y = int(point[0][1]*factor_h)
        points.append((x, y))
        # cv2.circle(original_img_copy, (x, y), 1, (255, 0 , 0), -1)
    cv2.polylines(original_img_copy, [np.array(points).reshape(-1, 1, 2)], True, (0, 0, 255), 1)
    # x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(original_img_copy, (int(x*factor_w), int(y*factor_h)),
    #               (int((x+w)*factor_w), int((y+h)*factor_h)),
    #               (255, 0, 0), 1)

cv2.imshow("original img", original_img)
cv2.imshow("contours", img)
cv2.imshow("bboxes", original_img_copy)
cv2.imwrite("results/contours_test/9_polylines.png", original_img_copy)
cv2.waitKey(0)