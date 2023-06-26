from word_segmentor_model import unet
from image_processing import pad_image
import matplotlib.pyplot as plt
import os
from text_extraction_model import inference_model
from text_extraction_utils import create_vocab, preprocess_img
import cv2
import numpy as np
import keras.backend


line_seg_model = unet(pretrained_weights="models/50.h5")
word_seg_model = unet(pretrained_weights="models/wordseg-20.h5")


def line_detection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = line_seg_model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    img = cv2.normalize(pred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)

    original_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # ori = original_img.copy()
    # ori = cv2.resize(ori, (512, 512))
    # cv2.imshow("img", img)
    # cv2.imshow("img original", ori)
    # cv2.waitKey(0)

    (h, w) = original_img.shape[:2]
    re_h, re_w = 512, 512
    factor_h, factor_w = h / float(re_h), w / float(re_w)

    coordinates = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if int(w * factor_w) >= 7 and int(h * factor_h) >= 5:
            coordinates.append((int(x * factor_w), int(y * factor_h), int((x + w) * factor_w), int((y + h) * factor_h)))

    count = 0

    for i in range(len(coordinates)):
        coord = coordinates[i]
        line_img = original_img[coord[1]:coord[3], coord[0]:coord[2]].copy()
        cv2.imwrite(f"results/line_images/{count}.jpg", line_img)
        count += 1

    return coordinates


def word_detection(path, line_coords):
    # image_list = os.listdir("results/line_images")
    # image_list = [file.split(".")[0] for file in image_list]
    img = cv2.imread(path)
    image_list = []
    for coord in line_coords:
        x1, y1, x2, y2 = coord
        line = img[y1:y2, x1:x2]
        line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        image_list.append(line)
    count1 = 0
    all_words_coordinates = []
    for image in image_list:
        img = image.copy()
        img = pad_image(img)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        pred = word_seg_model.predict(img)
        pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)
        plt.imsave(f"results/word_masks/{count1}_mask.jpg", pred)

        original_img = image.copy()
        img = cv2.imread(f"results/word_masks/{count1}_mask.jpg", cv2.IMREAD_GRAYSCALE)
        cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
        original_img = pad_image(original_img)
        (h, w) = original_img.shape[:2]
        factor_h, factor_w = h/512.0, w/512.0
        original_img_copy = np.stack((original_img,)*3, axis=-1)


        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        word_coordinates = []
        count = 0
        # print(f"{factor_w}    {w}        {factor_h}      {h}")
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # print([x, y, x+w, y+h])
            if int(w * factor_w) >= 7 and int(h * factor_h) >= 5:
                word_coordinates.append([int(x*factor_w), int(y*factor_h), int((x+w)*factor_w), int((y+h)*factor_h)])
                word = original_img[int(y*factor_h):int((y+h)*factor_h), int(x*factor_w):int((x+w)*factor_w)]
                cv2.imwrite(f"results/words/{count1}_{count}.png", word)
                count += 1

        count1 += 1
        all_words_coordinates.append(word_coordinates)
        # cv2.imwrite(f"results/word_masks/{image_path}_contours.png", original_img_copy)

    return all_words_coordinates


def text_inference(path, line_coords, all_words_coordinates):
    # vocab = create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")
    vocab = create_vocab("C:\\Users\\nexus\\PycharmProjects\\OCRDataset\\augementing\\augmented_FUNSD_texts")

    model = inference_model(input_dim=(32, 128, 1), output_dim=len(vocab))
    model.load_weights("models/text_model.hdf5")

    images = []
    word_sp = []
    # image_names = os.listdir("results/words")
    # for image in image_names:
    #     img = cv2.imread(f"results/words/{image}", cv2.IMREAD_GRAYSCALE)
    #     img = preprocess_img(img, (128, 32))
    #     img = np.expand_dims(img, axis=-1)
    #     img = img / 255
    #     images.append(img)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    width = img.shape[1]
    height = img.shape[0]

    assert len(line_coords) == len(all_words_coordinates)

    for i, line in enumerate(all_words_coordinates):
        for word_coords in line:
            print(word_coords)
            x1, y1, x2, y2 = word_coords
            line_x1, line_y1, line_x2, line_y2 = line_coords[i]
            # print(img.shape[0] >= line_y1+y2)
            # if not img.shape[0] >= line_y1 + y2:
            #     print(f"{img.shape[0]}       {line_y1}      {y2}")
            # print(img.shape[1] >= line_x1+x2)
            # if not img.shape[1] >= line_x1+x2:
            #     print(f"{img.shape[1]}         {line_x1}      {x2}")
            word = img[line_y1+y1:line_y1+y2, line_x1+x1:line_x1+x2]
            # cv2.imshow("word", word)
            # cv2.waitKey(0)
            word = preprocess_img(word, (128, 32))
            word = np.expand_dims(word, axis=-1)
            word = word / 255
            images.append(word)
            word_sp.append((line_x1+x1, line_y1+y1))

    images = np.array(images)

    prediction = model.predict(images)
    output = keras.backend.get_value(keras.backend.ctc_decode(
        prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True
    )[0][0])

    blank = np.zeros((3*height, 3*width))
    blank = 255-blank

    for i, p in enumerate(output):
        text = ""
        for x in p:
            if int(x) != -1:
                text += vocab[int(x)]
        print(text)
        cv2.putText(blank, text, 3*np.array(word_sp[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    return blank

def main():
    path = "dataset/LineSeg/354.JPG"
    line_coords = line_detection(path)
    all_words_coordinates = word_detection(path, line_coords)
    words_doc = text_inference(path, line_coords, all_words_coordinates)
    cv2.imwrite("results/doc_354.png", words_doc)



if __name__ == "__main__":
    main()
