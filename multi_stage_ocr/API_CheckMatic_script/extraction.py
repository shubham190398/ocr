import csv
import time
from paddleocr import PaddleOCR
import cv2
import torch.cuda
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from pypdfium2 import PdfDocument


DEVICE = torch.device('cuda')


def detection(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text


def get_text_region(image, ocr):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results = ocr.ocr(img, cls=True)
    return results


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


def reformat_into_rows(bboxes, row_sep):
    """
    takes a list of bboxes which are ordered according to their y-value and returns a
    row_dict, where bboxes are ordered according to their x-value within each row
    :param bboxes: ordered list of bboxes (according to y-value)
    :param row_sep: distance for separating rows
    :return: ordered row dict of bboxes
    """
    row_dict = {}
    last_row_y_position = -100
    last_row = 0

    for box in bboxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        if y1 - last_row_y_position > row_sep:
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


def get_text(rows, model, processor, image, minimum_spacing):
    text_dict = {}

    for key, values in rows.items():
        print(f"Extracting row {key}")
        text_dict[key] = []

        for value in values:
            x1, y1, x2, y2 = value[0][0], value[0][1], value[1][0], value[1][1]

            if abs(y1 - y2) > minimum_spacing and abs(x1 - x2) > minimum_spacing:
                crop = image[y1:y2, x1:x2]
                text_dict[key].append(detection(crop, model, processor))

    return text_dict


def write_csv(texts):
    with open('results/Bad Image 2.csv', 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)

        for key, value in texts.items():
            writer.writerow(value)


def invoice_transcribe(model, processor, ocr, invoice_file):
    results = get_text_region(invoice_file, ocr)
    rows = reformat_into_rows(reformat_boxes_to_be_usable_with_row_dict_func(results[0]), 7)
    texts = get_text(rows, model, processor, invoice_file, 7)
    write_csv(texts)


def transcribe(model, processor, ocr, pdf_file_path):
    pdf_file = PdfDocument(pdf_file_path)
    invoice_file = pdf_file[len(pdf_file) - 1]
    invoice_file = invoice_file.render(scale=2).to_numpy()
    invoice_transcribe(model, processor, ocr, invoice_file)


def main(pdf_file_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    model.to(DEVICE)

    transcribe(model, processor, ocr, pdf_file_path)


path = "../dataset/bad_img_pdfs/Bad Image 2.pdf"
t = time.time()
main(path)
elapsed = time.time() - t
with open('results/times.txt', 'a', encoding='UTF-8', newline='') as f:
    f.write(str(path) + '            time: ' + str(elapsed) + '\n')
