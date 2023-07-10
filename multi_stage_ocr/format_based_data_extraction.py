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
            'name': [144, 214, 1820, 320],
            'amount': [1757, 401, 2287, 499],
            'micr': [422, 926, 1930, 1050],
            'date': [1801, 87, 2275, 150],
            'data': 'handwritten'
        },
        'california': {
            'name': [246, 138, 1662, 432],
            'amount': [1698, 330, 2141, 446],
            'micr': [68, 871, 1401, 963],
            'date': [1439, 186, 1883, 297],
            'data': 'handwritten'
        },
        'canara': {
            'name': [200, 230, 1866, 328],
            'amount': [1841, 407, 2300, 521],
            'micr': [448, 940, 1920, 1046],
            'date': [1831, 98, 2300, 155],
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
            'name': [310, 216, 2108, 322],
            'amount': [1854, 396, 2322, 500],
            'micr': [462, 956, 1872, 1052],
            'date': [1835, 86, 2316, 149],
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
            'name': [184, 210, 1876, 314],
            'amount': [1821, 399, 2271, 492],
            'micr': [486, 966, 1862, 1052],
            'date': [1773, 84, 2301, 150],
            'data': 'handwritten'
        },
    }

    # assert dir == list(format.keys())

    for file in dir:
        if file not in ['uncategorised', 'hbl']:
            images_list = os.listdir(f'dataset/formats/{file}')
            for img_file in images_list:
                func = globals()[f"text_detector_{format[file]['data']}"]
                img = cv2.imread(f'dataset/formats/{file}/{img_file}')
                name = func(img[
                                format[file]['name'][1]:format[file]['name'][3],
                                format[file]['name'][0]:format[file]['name'][2]
                            ])
                cv2.imshow('name', img[
                                format[file]['name'][1]:format[file]['name'][3],
                                format[file]['name'][0]:format[file]['name'][2]
                            ])
                amount = func(img[
                                format[file]['amount'][1]:format[file]['amount'][3],
                                format[file]['amount'][0]:format[file]['amount'][2]
                            ])
                cv2.imshow('amount', img[
                                format[file]['amount'][1]:format[file]['amount'][3],
                                format[file]['amount'][0]:format[file]['amount'][2]
                            ])
                micr = text_detector_MICR(img[
                                format[file]['micr'][1]:format[file]['micr'][3],
                                format[file]['micr'][0]:format[file]['micr'][2]
                            ])
                cv2.imshow('micr', img[
                                format[file]['micr'][1]:format[file]['micr'][3],
                                format[file]['micr'][0]:format[file]['micr'][2]
                            ])
                date = func(img[
                                format[file]['date'][1]:format[file]['date'][3],
                                format[file]['date'][0]:format[file]['date'][2]
                            ])
                cv2.imshow('date', img[
                                format[file]['date'][1]:format[file]['date'][3],
                                format[file]['date'][0]:format[file]['date'][2]
                            ])
                cv2.waitKey(0)
                # f = open(f'results/format_results/{img_file.split(".")[0]}.txt', 'w')
                # micr_codes = []
                # for string in re.split('A|B|C|D|A |B |C |D ', micr):
                #     if string != '':
                #         full = ''
                #         for sub in string.split():
                #             full += sub
                #         micr_codes.append(full)
                # f.write('Name: ' + name + '\n' +
                #         'Amount: ' + amount + '\n' +
                #         'Date: ' + date + '\n' +
                #         'Cheque Number: ' + micr_codes[0] + '\n' +
                #         'Full MICR: ' + str([f'{micr_codes[i]} ' for i in range(len(micr_codes)-1)]) + micr_codes[-1])
                # f.close()


main()
