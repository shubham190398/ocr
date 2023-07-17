import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

def text_detector_printed(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

def text_detector_MICR(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("Apocalypse-19/trocr-MICR")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def main():
    dir = os.listdir('dataset/bad_images/micr')
    f = open('results/bad_images_micr.txt', 'w')
    for file in dir:
        path = 'dataset/bad_images/micr/' + file
        img = cv2.imread(path)
        text = text_detector_MICR(img)
        f.write(file + ', ' + text + '\n')
    f.close()

main()