
import cv2
import torch.cuda
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os


def detection(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text


DEVICE = torch.device('cuda')

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.to(DEVICE)

hdwrt = os.listdir('handwritten/images')
f = open('handwritten.txt', 'w')

for img_path in hdwrt:
    img = cv2.imread('handwritten/images/' + img_path)
    text = detection(img, model, processor)
    f.write(f'{img_path}                    {text}\n')

f.close()
