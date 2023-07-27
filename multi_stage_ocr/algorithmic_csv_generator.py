import easyocr
import cv2
from typing import Any


def crops(image_path: str, reader: Any) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = reader.readtext(image)
    print(results)


def main():
    image_path = "dataset/consolidated_remittances/3_B.png"
    reader = easyocr.Reader(['en'])
    crops(image_path, reader)


if __name__ == '__main__':
    main()
