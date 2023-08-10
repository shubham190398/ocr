import copy
import numpy as np
import cv2
from word_segmentor_model import unet
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sympy import Line2D, Point2D


def detect_lines(img, img_name):
    model = unet(pretrained_weights="models/text_seg_model.h5")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        if int(w * width_factor) >= 7 and int(h * height_factor) >= 5:
            coordinates.append((int(x * width_factor), int(y * height_factor), int((x + w) * width_factor),
                                int((y + h) * height_factor)))

    line_images = []
    count = 0
    for i in range(len(coordinates)):
        coords = coordinates[i]
        line_img = original_image[coords[1]:coords[3], coords[0]:coords[2]].copy()
        # cv2.imwrite(f'archive/demo_lines/{img_name}_{count}.png', line_img)
        count += 1
        line_images.append(line_img)

    return line_images, coordinates


def text_detector(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def main2(path, img_name, type_of_img):
    # path = 'dataset/demo_imgs/'
    # img_name = ''
    # f = open('dataset/clientdata/classification.txt')
    # dic = json.load(f)
    # print(list(dic.keys()))
    # for file in list(dic.keys()):
    #     path = 'dataset/clientdata/' + file
    #     img_name, _ = file.split(".")
    if type_of_img == "cheque":
        img = cv2.imread(path)
        img1 = img[:int((2 * img.shape[0]) / 10), int((8 * img.shape[1]) / 10):]
        # img2 = img[int((8 * img.shape[0]) / 10):]
        s1 = text_detector(img1)
        # s2 = text_detector(img2)
        f = open(f'archive/client_text/{img_name}.txt', 'w')
        f.write(s1 + "\n")
        f.close()

    elif type_of_img == "invoice":
        img = cv2.imread(path)
        imgs, coords = detect_lines(img, img_name)
        exhaust_coords = coords.copy()
        f = open(f'archive/client_text/{img_name}.txt', 'w')
        for coord in coords:
            if coord in exhaust_coords:
                x1, y1, x2, y2 = coord
                p1 = Point2D(x1, y1)
                p2 = Point2D(x2, y1)
                line = Line2D(p1, p2)
                removal = []
                for line_img, coord2 in zip(imgs, exhaust_coords):
                    if line.distance(Point2D(coord2[0], coord2[1])) <= 5:
                        removal.append((line_img, coord2))
                        s = text_detector(cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR))
                        f.write(s + ", ")
                for item in removal:
                    imgs.remove(item[0])
                    exhaust_coords.remove(item[1])
                f.write("\n")
        f.close()

    else:
        print("Please ensure type of img is either 'cheque' or 'invoice'")


def main():
    path = "dataset/clientdata/41.png"      # Path to image goes here
    img_name = (path.split('/')[-1]).split('.')[0]
    type_of_img = "cheque"                  # Type of img. Make sure this is only either "cheque" or "invoice"
    main2(path, img_name, type_of_img)


if __name__ == "__main__":
    main()