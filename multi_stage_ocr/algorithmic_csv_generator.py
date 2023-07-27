import easyocr
import cv2
from typing import Any, List, Tuple, Dict


def crops(image_path: str, reader: Any) -> List[Tuple]:
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


def main() -> None:
    image_path = "dataset/consolidated_remittances/3_B.png"
    reader = easyocr.Reader(['en'])
    results = crops(image_path, reader)
    rows = row_check(results)
    print(rows)


if __name__ == '__main__':
    main()
