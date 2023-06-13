import os
from image_processing import batch_generator
from line_segmentor_model import unet
import random
from keras.callbacks import ModelCheckpoint


def main():
    image_names = list(range(1, 379))
    dir_path = "dataset/LineSeg"
    random.shuffle(image_names)
    training_images = image_names[:int(0.85*len(image_names))]
    testing_images = image_names[int(0.85*len(image_names)):]
    model = unet()

    model_checkpoint = ModelCheckpoint('weights{epoch:8d}.h5', save_weights_only=True, period=5)
    model.fit_generator(
        batch_generator(dir_path, training_images, 32),
        epochs=50,
        steps_per_epoch=1000,
        validation_data=batch_generator(dir_path, testing_images, 8),
        validation_steps=400,
        callbacks=[model_checkpoint],
        shuffle=1
    )


if __name__ == "__main__":
    main()
