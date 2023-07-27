import easyocr
import cv2
from typing import Any, List, Tuple, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
import os

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")


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
        if y1 - last_row_y_position > 30:
            last_row += 1
            last_row_y_position = y1
            row_dict[last_row] = []
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])
        else:
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])

    return row_dict


def get_text(row_dict: Dict, image_path: str) -> Dict:
    image = cv2.imread(image_path)
    text_dict = {}
    for key, value in row_dict.items():
        print(f"Extracting row {key}")
        text_dict[key] = []
        for v in value:
            x1, y1 = v[0]
            x2, y2 = v[1]
            cropped_img = image[y1:y2, x1:x2]
            text_dict[key].append(text_detector(cropped_img))

    return text_dict


def text_detector(image: Any) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def get_csv(texts: Dict, name: str) -> None:
    with open(f"results/full_extraction/{name}.txt", "w") as f:
        for key, value in texts.items():
            text = "|".join(value)
            f.write(f"{text}\n")

    f.close()
    # text_file = pd.read_csv("results/full_extraction/temp.txt")
    # text_file.to_csv("results/full_extraction/BA_1.csv", index=None)


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
            break

    return img_list


def main() -> None:
    print('main entered')
    reader = easyocr.Reader(['en'])
    for file in get_A_images_from_cons_rem():
        print(file)
        image_path = "dataset/consolidated_remittances/" + file
        results = recognize_text(image_path, reader)
        rows = row_check(results)
        texts = get_text(rows, image_path)
        get_csv(texts, file.split('.')[0])


if __name__ == '__main__':
    main()
