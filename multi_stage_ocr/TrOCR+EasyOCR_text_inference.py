import os
import cv2
from bbox_extraction import bbox_extract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")


def text_detector_printed(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def main():

    img_dir = os.listdir('dataset/invoices')
    count = 1
    for file in img_dir:

        img = cv2.imread('dataset/invoices/' + file)
        # f = open('archive/invoice_TROCR+EasyOCR_results/text/' + str(count) + '.txt', 'w')
        print('popo')
        crop_count = 1
        for coords in bbox_extract(img):

            print('lll')
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            crop = img[y1:y2, x1:x2]
            cv2.imwrite('archive/invoice_TROCR+EasyOCR_results/crops/' + str(count) + '_' + str(crop_count) + '.png', crop)
            # f.write(f'({str(x1)}, {str(y1)}), ({str(x2)}, {str(y2)}), {text_detector_printed(crop)}\n')
            crop_count += 1

        # f.close()
        count += 1


def main2():

    img_dir = os.listdir('dataset/invoices')
    count = 1
    for file in img_dir:

        img = cv2.imread('dataset/invoices/' + file)
        bbox_img = img.copy()
        print('popo')
        for coords in bbox_extract(img):

            print('lll')
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite('archive/invoice_TROCR+EasyOCR_results/bbox_imgs/' + str(count) + '.png', bbox_img)

        count += 1


main2()
