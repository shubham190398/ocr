import easyocr
import cv2
from typing import Any, List, Tuple, Dict


def crops(image_path: str, reader: Any) -> List[Tuple]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = reader.readtext(image)
    return results


def row_check(results: List[Tuple]) -> Dict:
    row_dict = {}
    return row_dict


def main() -> None:
    image_path = "dataset/consolidated_remittances/3_B.png"
    reader = easyocr.Reader(['en'])
    crops(image_path, reader)


if __name__ == '__main__':
    main()
