import numpy as np
import pypdfium2 as pdfium
import cv2
import os
import easyocr
import time

import torch.cuda
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv

reader = easyocr.Reader(['en'])

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.to(torch.device('cuda'))

processor_MICR = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model_MICR = VisionEncoderDecoderModel.from_pretrained("Apocalypse-19/trocr-MICR")
model_MICR.to(torch.device('cuda'))


def recognize_text(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(image)
    return results


def row_check(results):
    row_dict = {}
    last_row_y_position = -100
    last_row = 0

    for detection in results:
        bbox = detection[0]
        x1, y1 = bbox[0]
        if isinstance(y1, float):
            x2, y2 = bbox[2]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            for key in row_dict.keys():
                if y1 - row_dict[key][0][0][1] <= 7:
                    for i, bbox1 in enumerate(row_dict[key]):
                        if bbox1[0][0] > x1:
                            row_dict[key].insert(i, [[x1, y1], [x2, y2]])
                            break
                    break
        else:
            if y1 - last_row_y_position > 7:
                last_row += 1
                last_row_y_position = y1
                row_dict[last_row] = []
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])

    return row_dict


def get_text(row_dict, image):
    text_dict = {}
    # img = image.copy()
    for key, value in row_dict.items():
        print(f"Extracting row {key}")
        text_dict[key] = []
        for v in value:
            x1, y1 = v[0]
            x2, y2 = v[1]
            if abs(y1 - y2) > 7 and abs(x1 - x2) > 7:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 0), 1)
                cropped_img = image[y1:y2, x1:x2]
                text_dict[key].append(text_detector(cropped_img))

    # cv2.imshow('ww', img)
    # cv2.waitKey(0)

    return text_dict


def get_text_and_cheque_number(row_dict, image):
    text_dict = {}
    cheque_number = ''
    last_norm = np.inf
    for key, value in row_dict.items():
        print(f"Extracting row {key}")
        text_dict[key] = []
        for v in value:
            x1, y1 = v[0]
            x2, y2 = v[1]
            p = (abs(((x1+x2)/2) - image.shape[1]), (y1+y2)/2)
            if abs(y1 - y2) > 7 and abs(x1 - x2) > 7:
                cropped_img = image[y1:y2, x1:x2]
                text = text_detector(cropped_img)
                text_dict[key].append(text)
                if text.replace('-', '').isnumeric() and not (y1 > last_norm or image.shape[1] - x1 > last_norm):
                    norm = np.linalg.norm(p)
                    if norm < last_norm:
                        cheque_number = text
                        last_norm = norm

    return cheque_number, text_dict


def text_detector(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(torch.device('cuda'))
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def text_detector_MICR(image):
    pixel_values = processor_MICR(image, return_tensors="pt").pixel_values.to(torch.device('cuda'))
    generated_ids = model_MICR.generate(pixel_values)
    generated_text = processor_MICR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def write_to_text(texts, name):
    with open('../result/' + name + '.txt', 'w') as f:
        for key, value in texts.items():
            text = ", ".join(value)
            f.write(f"{text}\n")

    f.close()


def write_to_csv(texts, name):
    with open('../result/' + name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for key, value in texts.items():
            writer.writerow(value)

    f.close()


def cheque_transcribe(img, name):

    results = recognize_text(img)
    rows = row_check(results)
    micr_bboxes = list(rows.values())[-1]
    rows = dict(list(rows.items())[:-1])
    # cheque_number, texts = get_text_and_cheque_number(rows, img)
    texts = get_text(rows, img)

    write_to_text(texts, name)
    cheque_txt = open('../result/' + name + '.txt', 'a')
    micr_texts = []
    for bbox in micr_bboxes:
        p1, p2 = bbox
        x1, y1 = p1
        x2, y2 = p2
        if abs(y1 - y2) > 7 and abs(x1 - x2) > 7:
            micr = img[y1:y2, x1:x2]
            gen_text = text_detector_MICR(micr)
            micr_texts.append(gen_text)
    micr_text = ''.join(micr_texts)
    cheque_txt.write(micr_text + '\n')
    cheque_txt.write('\n')
    cheque_txt.write('cheque number: ' + micr_texts[0].replace('A', '').replace('B', '').replace('C', '').replace('D', '') + '\n')
    cheque_txt.close()


def invoice_transcribe(img, name):
    results = recognize_text(img)
    rows = row_check(results)
    texts = get_text(rows, img)
    write_to_csv(texts, name)


def main():
    pdf_dir = os.listdir('../data')
    f = open('../result/times_total.txt', 'w')
    f_chq = open('../result/times_cheque.txt', 'w')
    f_inv = open('../result/times_invoice.txt', 'w')
    for file in pdf_dir:
        try:
            t = time.time()

            pdf = pdfium.PdfDocument('../data/' + file)

            cheque_pdf = pdf[0]
            invoice_pdf = pdf[len(pdf)-1]

            cheque = cheque_pdf.render(scale=2).to_numpy()
            invoice = invoice_pdf.render(scale=2).to_numpy()
            name = file.split('.')[0]
            t_chq = time.time()
            cheque_transcribe(cheque, name + '_cheque')
            f_chq.write(file + ': ' + str(time.time() - t_chq) + '\n')
            t_inv = time.time()
            invoice_transcribe(invoice, name + '_invoice')
            f_inv.write(file + ': ' + str(time.time() - t_inv) + '\n')
            print('Time taken:' + str(time.time() - t))
            f.write(file + ': ' + str(time.time() - t) + '\n')
        except:
            pass

    f.close()
    f_chq.close()
    f_inv.close()


if __name__ == '__main__':
    main()
