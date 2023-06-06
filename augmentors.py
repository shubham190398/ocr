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
        self.random_chance = random_chance
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self.random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    @randomness_decorator
    def __call__(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        # do the augmentation here
        return image, annotation


class RandomBrightness(Augmentor):
    def __init__(self, random_chance: float = 0.5, delta: int = 100, log_level: int = logging.INFO) -> None:
        super(RandomBrightness, self).__init__(random_chance, log_level)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self.delta = delta

    @randomness_decorator
    def __call__(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        value = 1 + np.random.uniform(-self.delta, self.delta)/255
        hsv = np.array(image.HSV(), dtype=np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv = np.uint8(np.clip(hsv, 0, 255))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image.update(img)

        return image, annotation


class RandomRotate(Augmentor):
    def __init__(
            self,
            random_chance: float = 0.5,
            angle: typing.Union[int, typing.List] = 30,
            bordervalue: typing.Tuple[int, int, int] = None,
            log_level: int = logging.INFO,
    ) -> None:
        super(RandomRotate, self).__init__(random_chance, log_level)
        self.angle = angle
        self.bordervalue = bordervalue

    @randomness_decorator
    def __call__(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        if isinstance(self.angle, list):
            angle = float(np.random.choice(self.angle))
        else:
            angle = float(np.random.uniform(-self.angle, self.angle))

        bordervalue = np.random.randint(0, 255, 3) if self.bordervalue is None else self.bordervalue
        bordervalue = [int(val) for val in bordervalue]

        center_x, center_y = image.center
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_width = int((image.height * sin) + (image.width * cos))
        new_height = int((image.height * cos) + (image.width * sin))

        M[0, 2] += (new_width / 2) - center_x
        M[1, 2] += (new_height / 2) - center_y

        img = cv2.warpAffine(image.numpy(), M, (new_width, new_height), borderValue=bordervalue)
        image.update(img)

        return image, annotation


class RandomErodeDilate(Augmentor):
    def __init__(
        self,
        random_chance: float = 0.5,
        kernel_size: typing.Tuple[int, int] = (1, 1),
        log_level: int = logging.INFO,
    ) -> None:
        super(RandomErodeDilate, self).__init__(random_chance, log_level)
        self.kernel_size = kernel_size

    @randomness_decorator
    def __call__(self, image: NormalImage, annotation: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        kernel = np.ones(self.kernel_size, np.uint8)

        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), kernel, iterations=1)

        image.update(img)

        return image, annotation
