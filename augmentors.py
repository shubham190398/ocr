import cv2
import typing
import numpy as np
import logging
from image_formats import NormalImage


def randomness_decorator(func):
    def wrapper(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        if not isinstance(image, NormalImage):
            self.logger.error(f"image must be Image object, not {type(image)}, skipping augmentor")
            return image, annotation

        if np.random.rand() > self._random_chance:
            return image, annotation

        return func(self, image, annotation)

    return wrapper


class Augmentor:
    def __init__(self, random_chance: float = 0.5, log_level: int = logging.INFO) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    @randomness_decorator
    def __call__(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        # do the augmentation here
        return image, annotation

