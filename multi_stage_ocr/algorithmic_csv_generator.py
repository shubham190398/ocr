import easyocr
import cv2
from typing import Any, List, Tuple, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd

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
            row_dict[last_row].append([bbox[0], bbox[2]])
        else:
            row_dict[last_row].append([bbox[0], bbox[2]])

    return row_dict


def get_text(row_dict: Dict, image_path: str) -> Dict:
    image = cv2.imread(image_path)
    text_dict = {}
    for key, value in row_dict.items():
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


def get_csv(texts: Dict) -> None:
    with open("results/full_extraction/temp.txt", "w") as f:
        for key, value in texts.items():
            text = ",".join(value)
            f.write(f"{text}\n")

    text_file = pd.read_csv("results/full_extraction/temp.txt")
    text_file.to_csv("results/full_extraction/BA_1.csv", index=None)


def main() -> None:
    image_path = "dataset/consolidated_remittances/BA - 1.jpeg"
    reader = easyocr.Reader(['en'])
    results = recognize_text(image_path, reader)
    rows = row_check(results)
    texts = get_text(rows, image_path)
    get_csv(texts)


if __name__ == '__main__':
    main()
