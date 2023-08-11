import pypdfium2 as pdfium
import cv2
import os
import easyocr
import time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv

reader = easyocr.Reader(['en'])

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

processor_MICR = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model_MICR = VisionEncoderDecoderModel.from_pretrained("Apocalypse-19/trocr-MICR")


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
        if y1 - last_row_y_position > 5:
            last_row += 1
            last_row_y_position = y1
            row_dict[last_row] = []
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])
        else:
            row_dict[last_row].append([[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[2][0]), int(bbox[2][1])]])

    return row_dict


def get_text(row_dict, image):
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


def text_detector(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def text_detector_MICR(image):
    pixel_values = processor_MICR(image, return_tensors="pt").pixel_values
    generated_ids = model_MICR.generate(pixel_values)
    generated_text = processor_MICR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def write_to_text(texts, name):
    with open('../results/cheque+invoice_full_extraction/' + name + '.txt', 'w') as f:
        for key, value in texts.items():
            text = ", ".join(value)
            f.write(f"{text}\n")

    f.close()


def write_to_csv(texts, name):
    with open('../results/cheque+invoice_full_extraction/' + name + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for key, value in texts.items():
            writer.writerow(value)

    f.close()


def cheque_transcribe(img, name):

    results = recognize_text(img)
    rows = row_check(results)
    micr_bboxes = list(rows.values())[-1]
    rows = dict(list(rows.items())[:-1])
    texts = get_text(rows, img)
    write_to_text(texts, name)
    cheque_txt = open('../results/cheque+invoice_full_extraction/' + name + '.txt', 'a')
    micr_texts = []
    for bbox in micr_bboxes:
        p1, p2 = bbox
        cv2.rectangle(img, p1, p2, (250, 0, 0), 1)
        x1, y1 = p1
        x2, y2 = p2
        micr = img[y1:y2, x1:x2]
        gen_text = text_detector_MICR(micr)
        micr_texts.append(gen_text)
    cv2.imshow('che', img)
    cv2.waitKey(0)
    micr_text = ''.join(micr_texts)
    cheque_txt.write(micr_text + '\n')
    cheque_txt.close()


def invoice_transcribe(img, name):
    results = recognize_text(img)
    rows = row_check(results)
    print('invoice\n')
    print(results)
    texts = get_text(rows, img)
    write_to_csv(texts, name)


def main():
    pdf_dir = os.listdir('../dataset/bad_img_pdfs')

    for file in pdf_dir:
        # t = time.time()
        #
        # pdf = pdfium.PdfDocument('../dataset/bad_img_pdfs/' + file)
        # print(len(pdf))
        #
        # cheque_pdf = pdf[0]
        # invoice_pdf = pdf[len(pdf)-1]
        #
        # cheque = cheque_pdf.render(scale=2).to_numpy()
        # invoice = invoice_pdf.render(scale=2).to_numpy()
        # name = file.split('.')[0]
        # cheque_transcribe(cheque, name + '_cheque')
        # invoice_transcribe(invoice, name + '_invoice')
        # print('Time taken:' + str(time.time() - t))
        if file == 'Bad Image 2.pdf':
            t = time.time()

            pdf = pdfium.PdfDocument('../dataset/bad_img_pdfs/' + file)
            print(len(pdf))

            cheque_pdf = pdf[0]
            invoice_pdf = pdf[len(pdf) - 1]

            cheque = cheque_pdf.render(scale=2).to_numpy()
            invoice = invoice_pdf.render(scale=2).to_numpy()
            name = file.split('.')[0]
            cheque_transcribe(cheque, name + '_cheque')
            invoice_transcribe(invoice, name + '_invoice')
            print('Time taken:' + str(time.time() - t))


if __name__ == '__main__':
    main()
