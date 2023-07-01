import copy
import numpy as np
import cv2
from word_segmentor_model import unet
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


def detect_lines(image_path):
    model = unet(pretrained_weights="models/50.h5")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = img.shape[:2]
    original_image = copy.deepcopy(img)

    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predictions = np.squeeze(np.squeeze(predictions, axis=0), axis=-1)

    coordinates = []
    img = cv2.normalize(src=predictions, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)

    width_factor, height_factor = original_width/float(512), original_height/float(512)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        coordinates.append((int(x * width_factor), int(y * height_factor), int((x + w) * width_factor),
                            int((y + h) * height_factor)))

    line_images = []

    for i in range(len(coordinates)):
        coords = coordinates[i]
        line_img = original_image[coords[1]:coords[3], coords[0]:coords[2]].copy()
        line_images.append(line_img)

    return line_images, coordinates


def text_detector(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


def main():
    image = Image.open("results/line_images/82250337_0338_24.jpg").convert("RGB")
    text_detector(image)


if __name__ == "__main__":
    main()
