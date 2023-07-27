import easyocr
import cv2

image_path = 'demo_imgs2/215.JPG'

reader = easyocr.Reader(['en'])
image = cv2.imread(image_path)
num = 600
Result = reader.readtext(image)
for i in Result:
    num += 1
    r, text, conf = i
    r1, r2, r3, r4 = r
    x1, y1 = r1
    x2, y2 = r3
    print(x1, y1, x2, y2)
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(f'crops_easy_hand/{num}.png', crop)
    cv2.waitKey(0)
    print(i)
