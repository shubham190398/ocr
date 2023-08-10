import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def trocr_base_printed(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text
def trocr_base_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

def trocr_large_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

def dunnbc22_trocr_base_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

def thivy_trocr_large_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("thivy/num-trocr-large-handwritten-v1")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


dir = os.listdir('dataset/handwritten_cheques')

base_p = open('archive/handwritten_results/trocr_base_printed.txt', 'a')
base_h = open('archive/handwritten_results/trocr_base_handwritten.txt', 'a')
large_h = open('archive/handwritten_results/trocr_large_handwritten.txt', 'a')
dunnbc22 = open('archive/handwritten_results/dunnbc22_trocr_large_handwritten.txt', 'a')
thivy = open('archive/handwritten_results/thivy_trocr_large_handwritten.txt', 'a')

for file in dir:
    path = 'dataset/handwritten_cheques/' + file
    if file.split('.')[0] in ['1', '1a', '1b', '1c', '2', '2a']:
        img = cv2.imread(path)
        base_p.write(f'{file}, {trocr_base_printed(img)}\n')
        base_h.write(f'{file}, {trocr_base_handwritten(img)}\n')
        large_h.write(f'{file}, {trocr_large_handwritten(img)}\n')
        dunnbc22.write(f'{file}, {dunnbc22_trocr_base_handwritten(img)}\n')
        thivy.write(f'{file}, {thivy_trocr_large_handwritten(img)}\n')

base_p.close()
base_h.close()
large_h.close()
dunnbc22.close()
thivy.close()
