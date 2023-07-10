import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from random import sample
import numpy as np

IMG_HEIGHT = 1050
IMG_WIDTH = 2300

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
        # 'a': {
        #     'name': [336, 340, 1754, 406],
        #     'amount': [1835, 342, 2109, 400],
        #     'micr': [339, 781, 1624, 848],
        #     'date': [1542, 213, 1902, 309],
        #     'data': 'printed'
        # },
        'axis': {
            'name': [140, 190, 1769, 311],
            'amount': [1708, 380, 2224, 510],
            'micr': [410, 901, 1876, 1021],
            'date': [1760, 80, 2220, 145],
            'data': 'handwritten'
        },
        'california': {
            'name': [353, 310, 1650, 475],
            'amount': [1750, 340, 2198, 480],
            'micr': [70, 867, 1450, 963],
            'date': [1407, 180, 1933, 325],
            'data': 'handwritten'
        },
        'canara': {
            'name': [202, 224, 1822, 308],
            'amount': [1795, 353, 2231, 526],
            'micr': [405, 888, 1860, 1002],
            'date': [1767, 85, 2245, 159],
            'data': 'handwritten'
        },
        'f': {
            'name': [274, 548, 811, 609],
            'amount': [1739, 481, 2104, 537],
            'micr': [234, 842, 2176, 922],
            'date': [1766, 200, 2125, 268],
            'data': 'printed'
        },
        'g': {
            'name': [287, 366, 849, 433],
            'amount': [1909, 415, 2149, 462],
            'micr': [382, 830, 1754, 900],
            'date': [1499, 415, 1749, 465],
            'data': 'printed'
        },
        'i': {
            'name': [405, 633, 779, 688],
            'amount': [1737, 331, 2129, 410],
            'micr': [248, 872, 1532, 948],
            'date': [1722, 205, 1931, 245],
            'data': 'printed'
        },
        'icici': {
            'name': [300, 206, 2108, 304],
            'amount': [1787, 369, 2233, 502],
            'micr': [462, 909, 1856, 1024],
            'date': [1767, 76, 2232, 152],
            'data': 'handwritten'
        },
        'j': {
            'name': [151, 527, 1175, 586],
            'amount': [1446, 369, 1711, 417],
            'micr': [486, 774, 1744, 822],
            'date': [849, 110, 1366, 162],
            'data': 'printed'
        },
        # 'js': {
        #     'name': [],
        #     'amount': [],
        #     'micr': [],
        #     'date': [],
        #     'data': 'handwritten'
        # },
        'k': {
            'name': [280, 588, 474, 634],
            'amount': [1883, 344, 2180, 422],
            'micr': [333, 819, 1503, 897],
            'date': [1611, 233, 2005, 296],
            'data': 'printed'
        },
        'syndicate': {
            'name': [188, 200, 1824, 312],
            'amount': [1766, 366, 2201, 492],
            'micr': [441, 915, 1797, 1020],
            'date': [1719, 67, 2231, 152],
            'data': 'handwritten'
        },
    }

# CONVERT VALUES TO PERCENTAGE OF WIDTH AND HEIGHT AT THE END
    for file in dir:
        print(file)
        if file == 'canara': #not in ['uncategorised', 'hbl', 'a', 'b', 'd', 'e', 'h', 'js', 'meezan']:
            images_list = os.listdir(f'dataset/cheque_formats/{file}')
            # images_list = sample(images_list, 5 if 5 < len(images_list) else len(images_list))
            for img_file in images_list:
                print(img_file)
                func = globals()[f"text_detector_{format[file]['data']}"]
                img = cv2.imread(f'dataset/cheque_formats/{file}/{img_file}')
                height = img.shape[0]
                width = img.shape[1]
                name = func(img[
                                format[file]['name'][1]:format[file]['name'][3],
                                format[file]['name'][0]:format[file]['name'][2]
                            ])
                cv2.imshow('name', img[
                                format[file]['name'][1]:format[file]['name'][3],
                                format[file]['name'][0]:format[file]['name'][2]
                            ])
                amount = text_detector_printed(img[
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
                date = text_detector_printed(img[
                                format[file]['date'][1]:format[file]['date'][3],
                                format[file]['date'][0]:format[file]['date'][2]
                            ])
                cv2.imshow('date', img[
                                format[file]['date'][1]:format[file]['date'][3],
                                format[file]['date'][0]:format[file]['date'][2]
                            ])
                cv2.waitKey(0)
                # f = open(f'results/format_results/{img_file.split(".")[0]}.txt', 'w')
                # proper_date = ''
                # for char in date:
                #     if char in ['0', '1', '2', '3', '4', '5', '6',' 7', '8', '9']:
                #         proper_date += char
                # proper_date = proper_date[:1] + '-' + proper_date[2:3] + '-' + proper_date[4:] if len(proper_date) == 7 else date
                # micr_codes = []
                # for string in re.split('A|B|C|D|A |B |C |D ', micr):
                #     if string != '':
                #         full = ''
                #         for sub in string.split():
                #             full += sub
                #         micr_codes.append(full)
                # f.write('Name: ' + name + '\n' +
                #         'Amount: ' + amount + '\n' +
                #         'Date: ' + proper_date + '\n' +
                #         'Cheque Number: ' + micr_codes[0] + '\n' +
                #         'Full MICR: ' + str([f'{micr_codes[i]} ' for i in range(len(micr_codes)-1)]) + micr_codes[-1])
                # f.close()


main()
