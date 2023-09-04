import csv
import time

import cv2
import torch.cuda
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from easyocr import Reader
from pypdfium2 import PdfDocument


DEVICE = torch.device('cuda')


def detection(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text


def get_text_region(image, reader):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(img)
    return results


def get_rows(results, row_separation):
    row_dict = {}
    row_count = 0
    last_row_position = -100

    for result in results:
        bbox = result[0]
        y1 = bbox[0][1]

        if isinstance(y1, float):
            x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])

            for key in row_dict.keys():
                if y1 - row_dict[key][0][0][1] <= row_separation:
                    for i, bbox2 in enumerate(row_dict[key]):
                        if x1 < bbox2[0][0]:
                            row_dict[key].insert(i, [[x1, y1], [x2, y2]])
                            break
                    break

        else:
            if y1 - last_row_position > row_separation:
                row_count += 1
                last_row_position = y1
                row_dict[row_count] = []

            x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
            row_dict[row_count].append([[x1, y1], [x2, y2]])

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


def invoice_transcribe(model, processor, reader, invoice_file):
    results = get_text_region(invoice_file, reader)
    rows = get_rows(results, 7)
    texts = get_text(rows, model, processor, invoice_file, 7)
    write_csv(texts)


def transcribe(model, processor, reader, pdf_file_path):
    pdf_file = PdfDocument(pdf_file_path)
    invoice_file = pdf_file[len(pdf_file) - 1]
    invoice_file = invoice_file.render(scale=2).to_numpy()
    invoice_transcribe(model, processor, reader, invoice_file)


def main(pdf_file_path):
    reader = Reader(['en'])

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    model.to(DEVICE)

    transcribe(model, processor, reader, pdf_file_path)


path = "../dataset/bad_img_pdfs/Bad Image 2.pdf"
t = time.time()
main(path)
elapsed = time.time() - t
with open('results/times.txt', 'a', encoding='UTF-8', newline='') as f:
    f.write(str(path) + '            time: ' + str(elapsed) + '\n')
