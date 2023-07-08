import cv2
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


direc = os.listdir("dataset/manual_crops2/a")

def text_detector(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text

for file in direc:
    sep = file.split("_")
    if len(sep) == 3:
        doc_num, _, end_ind = sep
    else:
        doc_num, end_ind = sep

    end_ind, _ = end_ind.split(".")
    end = (end_ind == "e")
    f = open(f"results/manual_crops_text/{doc_num}.txt", "a")
    if "invoice" in doc_num and not end:
        img = cv2.imread(f"dataset/manual_crops2/a/{file}")
        s = text_detector(img)
        f.write(s + ", ")
        f.close()
    else:
        img = cv2.imread(f"dataset/manual_crops2/a/{file}")
        s = text_detector(img)
        f.write(s + "\n")
        f.close()