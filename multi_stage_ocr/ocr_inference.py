import os
from text_extraction_model import inference_model
from text_extraction_utils import create_vocab, preprocess_img
import cv2
import numpy as np
import keras.backend


vocab = create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")

model = inference_model(input_dim=(32, 128, 1), output_dim=len(vocab))
model.load_weights("models/text_model.hdf5")

images = []
image_names = os.listdir("dataset/OCR")
for image in image_names:
    img = cv2.imread(f"dataset/OCR/{image}", cv2.IMREAD_GRAYSCALE)
    img = preprocess_img(img, (128, 32))
    img = np.expand_dims(img, axis=-1)
    img = img/255
    images.append(img)

images = np.array(images)

prediction = model.predict(images)
output = keras.backend.get_value(keras.backend.ctc_decode(
    prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True
)[0][0])

for p in output:
    text = ""
    for x in p:
        if int(x) != -1:
            text += vocab[int(x)]
    print(text)
