import cv2
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


direc = os.listdir("dataset/manual_crops")

def text_detector(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

for file in direc:
    _, ext = file.split(".")
    doc_num, _ = file.split("_")
    f = open(f"results/manual_crops_text/{doc_num}.txt", "a")
    img = cv2.imread(f"dataset/manual_crops/{file}")
    s = text_detector(img)
    f.write(s + "\n")
    f.close()