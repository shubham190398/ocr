import time

import easyocr
import cv2
from typing import Any, List, Tuple, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
import os

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
processor_base = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")


def recognize_text(image_path: str, reader: Any) -> List[Tuple]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = reader.readtext(image)
    return results


def row_check(results: List[Tuple]) -> Dict:
    row_dict = {}
    last_row_y_position = -100
    last_row = 0

    for detection in results:
        bbox = detection[0]
        x1, y1 = bbox[0]
        if y1 - last_row_y_position > 5:
            last_row += 1
            last_row_y_position = y1
            row_dict[last_row] = []
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])
        else:
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])

    return row_dict


def get_text(row_dict: Dict, image_path: str, text_detector_choice: Any) -> Dict:
    image = cv2.imread(image_path)
    text_dict = {}
    f = open('results/full_extraction/time_for_TROCR_word.txt', 'a')
    for key, value in row_dict.items():
        print(f"Extracting row {key}")
        text_dict[key] = []
        for v in value:
            x1, y1 = v[0]
            x2, y2 = v[1]
            cropped_img = image[y1:y2, x1:x2]
            t = time.time()
            text_dict[key].append(text_detector_choice(cropped_img))
            f.write(str(time.time() - t) + '\n')

    f.close()

    return text_dict


def text_detector(image: Any) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def text_detector_base(image: Any) -> str:
    pixel_values = processor_base(image, return_tensors="pt").pixel_values
    generated_ids = model_base.generate(pixel_values)
    generated_text = processor_base.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def get_csv(texts: Dict, name: str) -> None:
    with open(f"results/full_extraction/{name}.txt", "w") as f:
        for key, value in texts.items():
            text = "|".join(value)
            f.write(f"{text}\n")

    f.close()
    # text_file = pd.read_csv("results/full_extraction/temp.txt")
    # text_file.to_csv("results/full_extraction/BA_1.csv", index=None)


def get_text_from_EasyOCR(image_path: str, reader: Any):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = reader.readtext(image)
    row_dict = {}
    text_dict = {}
    last_row_y_position = -100
    last_row = 0

    for detection in results:
        bbox = detection[0]
        x1, y1 = bbox[0]
        if y1 - last_row_y_position > 5:
            last_row += 1
            last_row_y_position = y1
            row_dict[last_row] = []
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])
            text_dict[last_row] = []
            text_dict[last_row].append(detection[1])
        else:
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])
            text_dict[last_row].append(detection[1])

    return row_dict, text_dict


def get_A_images_from_cons_rem() -> List:
    print('get_img func entered')
    img_dir = os.listdir('dataset/consolidated_remittances')
    count = 1
    img_list = []
    for file in img_dir:
        if '_A.png' in file:
            cv2.imshow('check', cv2.imread('dataset/consolidated_remittances/' + file))
            key = cv2.waitKey(0)
            if key == ord('y'):
                img_list.append(file)
                count += 1
            if count >= 40:
                break

    return img_list


def get_B_images_from_cons_rem() -> List:
    print('get_img func entered')
    img_dir = os.listdir('dataset/consolidated_remittances')
    count = 1
    img_list = []
    for file in img_dir:
        if '_B.png' in file:
            cv2.imshow('check', cv2.imread('dataset/consolidated_remittances/' + file))
            key = cv2.waitKey(0)
            if key == ord('y'):
                img_list.append(file)
                count += 1
            if count >= 40:
                break

    return img_list


def get_bad_images_from_cons_rem() -> List:
    print('get_img func entered')
    img_dir = os.listdir('dataset/bad_images/invoices')
    img_list = []
    for file in img_dir:
        img_list.append(file)

    return img_list


def main() -> None:
    print('main entered')
    reader = easyocr.Reader(['en'])
    f = open('results/full_extraction/times_for_bad_images.txt', 'w')

    for file in get_bad_images_from_cons_rem():
        print(file)
        t = time.time()
        if int(file.split('.')[0].split('_')[1]) < 6:
            image_path = "dataset/bad_images/invoices/" + file
            results = recognize_text(image_path, reader)
            rows = row_check(results)
            texts = get_text(rows, image_path, text_detector)
            get_csv(texts, file.split('.')[0])
        else:
            image_path = "dataset/bad_images/invoices/" + file
            rows, texts = get_text_from_EasyOCR(image_path, reader)
            get_csv(texts, file.split('.')[0])
        print('Time taken for' + file + ' is ' + str(time.time() - t) + '. Rows = ' + str(max(list(rows.keys()))))
        f.write('Time taken for' + file + ' is ' + str(time.time() - t) + '. Rows = ' + str(max(list(rows.keys()))) + '\n')
    f.close()


def main2() -> None:
    print('main2 entered')
    reader = easyocr.Reader(['en'])

    for file in get_bad_images_from_cons_rem():
        print(file)
        if int(file.split('.')[0].split('_')[1]) < 6:
            image_path = "dataset/bad_images/invoices/" + file
            results = recognize_text(image_path, reader)
            rows = row_check(results)
            texts = get_text(rows, image_path, text_detector)
            get_csv(texts, file.split('.')[0])
        else:
            pass

    for file in get_bad_images_from_cons_rem():
        print(file)
        if int(file.split('.')[0].split('_')[1]) < 6:
            image_path = "dataset/bad_images/invoices/" + file
            results = recognize_text(image_path, reader)
            rows = row_check(results)
            texts = get_text(rows, image_path, text_detector_base)
            get_csv(texts, file.split('.')[0])
        else:
            pass



if __name__ == '__main__':
    main2()
