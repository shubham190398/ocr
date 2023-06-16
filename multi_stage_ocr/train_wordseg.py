from word_segmentor_model import batch_segmentor, unet
import random
from keras.callbacks import ModelCheckpoint
import os


def main():
    dir_path = ""
    image_names = os.listdir(dir_path)
    image_names = [filename.split(".")[0] for filename in image_names]
    random.shuffle(image_names)
    training_images = image_names[:int(0.9*len(image_names))]
    testing_images = image_names[int(0.9*len(image_names)):]
    model = unet()

    model_checkpoint = ModelCheckpoint('weights{epoch:8d}.h5', save_weights_only=True, period=5)
    model.fit_generator(
        batch_segmentor(dir_path, training_images, 16),
        epochs=60,
        steps_per_epoch=1000,
        validation_data=batch_segmentor(dir_path, testing_images, 8),
        validation_steps=400,
        callbacks=[model_checkpoint],
        shuffle=1
    )


if __name__ == "__main__":
    main()
