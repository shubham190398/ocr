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
    processor = TrOCRProcessor.from_pretrained("DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2")
    model = VisionEncoderDecoderModel.from_pretrained("DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

def thivy_trocr_large_handwritten(image):
    processor = TrOCRProcessor.from_pretrained("thivy/num-trocr-large-handwritten-v1")
    model = VisionEncoderDecoderModel.from_pretrained("thivy/num-trocr-large-handwritten-v1")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


dir = os.listdir('dataset/handwritten_cheques')

base_p = open('results/handwritten_results/trocr_base_printed.txt', 'w')
base_h = open('results/handwritten_results/trocr_base_handwritten.txt', 'w')
large_h = open('results/handwritten_results/trocr_large_handwritten.txt', 'w')
dunnbc22 = open('results/handwritten_results/dunnbc22_trocr_large_handwritten.txt', 'w')
thivy = open('results/handwritten_results/thivy_trocr_large_handwritten.txt', 'w')

for file in dir:
    path = 'dataset/handwritten_cheques/' + file
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
