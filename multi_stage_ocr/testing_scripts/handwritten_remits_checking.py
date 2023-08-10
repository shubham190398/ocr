import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

def text_detector_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def main():
    dir = os.listdir('../dataset/handwritten_remits')
    f = open('../archive/handwritten_remits_text.txt', 'w')
    for file in dir:
        path = '../dataset/handwritten_remits/' + file
        img = cv2.imread(path)
        text = text_detector_handwritten(img)
        f.write(file + ', ' + text + '\n')
    f.close()

main()