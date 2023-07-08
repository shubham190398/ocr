import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def text_detector(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("Apocalypse-19/trocr-MICR")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

dir = os.listdir('dataset/micr')
f = open('results/micr.txt', 'w')

for file in dir:
    img = cv2.imread('dataset/micr/'+file)
    text = text_detector(img)
    f.write(f'{file}, {text}\n')

f.close()