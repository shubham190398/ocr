import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re


def text_detector_MICR(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("Apocalypse-19/trocr-MICR")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def text_detector_printed(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def text_detector_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def main():
    dir = os.listdir('dataset/cheque_formats')
    format = {
        'axis': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'california': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'canara': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'hbl': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'icici': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'js': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
        'syndicate': {
            'name': [],
            'amount': [],
            'micr': [],
            'date': [],
            'data': 'handwritten'
        },
    }

    # assert dir == list(format.keys())

    for file in dir:
        if not file == 'uncategorised':
            images_list = os.listdir(f'dataset/formats/{file}')
            for img_file in images_list:
                func = globals()[f"text_detector_{format[file]['data']}"]
                img = cv2.imread(f'dataset/formats/{file}/{img_file}')
                name = func(img[
                                format[file]['name'][1]:format[file]['name'][3],
                                format[file]['name'][0]:format[file]['name'][2]
                            ])
                amount = func(img[
                                format[file]['amount'][1]:format[file]['amount'][3],
                                format[file]['amount'][0]:format[file]['amount'][2]
                            ])
                micr = text_detector_MICR(img[
                                format[file]['micr'][1]:format[file]['micr'][3],
                                format[file]['micr'][0]:format[file]['micr'][2]
                            ])
                date = func(img[
                                format[file]['date'][1]:format[file]['date'][3],
                                format[file]['date'][0]:format[file]['date'][2]
                            ])
                f = open(f'results/format_results/{img_file.split(".")[0]}.txt', 'w')
                micr_codes = []
                for string in re.split('A|B|C|D|A |B |C |D ', micr):
                    if string != '':
                        full = ''
                        for sub in string.split():
                            full += sub
                        micr_codes.append(full)
                f.write('Name: ' + name + '\n' +
                        'Amount: ' + amount + '\n' +
                        'Date: ' + date + '\n' +
                        'Cheque Number: ' + micr_codes[0] + '\n' +
                        'Full MICR: ' + str([f'{micr_codes[i]} ' for i in range(len(micr_codes)-1)]) + micr_codes[-1])
                f.close()


main()
