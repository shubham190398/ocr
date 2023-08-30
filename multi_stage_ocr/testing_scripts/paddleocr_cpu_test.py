import time

from paddleocr import PaddleOCR
import cv2
import numpy as np
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory


def reformat_boxes_to_be_usable_with_row_dict_func(result):
    """
    given the paddleocr result of an image, this function will return a list of bboxes
    ordered according to their y1 value (i.e. top-left corner y-value)
    :param result: paddle ocr result for an image
    :return: ordered list of bboxes
    """
    boxes = [(int(line[0][0][1]), line[0]) for line in result]

    dtype = [('height', int), ('list', list)]
    bboxes_with_height = np.array(boxes, dtype)
    bboxes_sorted = np.sort(bboxes_with_height, order='height')
    bboxes = []
    for _, bbox in bboxes_sorted:
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[2][0])
        y2 = int(bbox[2][1])
        bboxes.append([(x1, y1), (x2, y2)])

    return bboxes


def reformat_into_rows(bboxes):
    """
    takes a list of bboxes which are ordered according to their y-value and returns a
    row_dict, where bboxes are ordered according to their x-value within each row
    :param bboxes: ordered list of bboxes (according to y-value)
    :return: ordered row dict of bboxes
    """
    row_dict = {}
    last_row_y_position = -100
    last_row = 0

    for box in bboxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        if y1 - last_row_y_position > 20:
            last_row += 1
            last_row_y_position = y1
            row_dict[last_row] = []
        row_dict[last_row].append((x1, [[x1, y1], [x2, y2]]))

    for key, value in row_dict.items():
        dtype = [('x', int), ('list', list)]
        values_to_sort = np.array(value, dtype)
        values_sorted = np.sort(values_to_sort, order='x')
        new_value = []
        for _, bbox in values_sorted:
            new_value.append(bbox)
        row_dict[key] = new_value

    print(row_dict)
    return row_dict


img_path = '../dataset/consolidated_remittances/3_B.png'
t = time.time()
result = ocr.ocr(img_path, cls=True)
print(str(time.time() - t))
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

result = result[0]
img = cv2.imread(img_path)
boxes = [line[0] for line in result]
for bbox in boxes:
    x1 = int(bbox[0][0])
    y1 = int(bbox[0][1])
    x2 = int(bbox[2][0])
    y2 = int(bbox[2][1])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imwrite('result.png', img)