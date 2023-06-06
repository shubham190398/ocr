import cv2
import typing
import numpy as np
import logging
from image_formats import NormalImage

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Transformer:
    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError


class ExpandDims(Transformer):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    # noinspection PyMethodOverriding
    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label


class ImageResizer(Transformer):
    def __init__(
            self,
            width: int,
            height: int,
            keep_aspect_ratio: bool = False,
            padding_color: typing.Tuple[int] = (0, 0, 0)
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.padding_color = padding_color

    @staticmethod
    def resize_maintaining_aspect_ratio(
            image: np.ndarray,
            width_target: int,
            height_target: int,
            padding_color: typing.Tuple[int] = (0, 0, 0)
    ) -> np.ndarray:
        height, width = image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=padding_color)

        return new_image

    # noinspection PyMethodOverriding
    def __call__(self, image: NormalImage, label: typing.Any) -> typing.Tuple[NormalImage, typing.Any]:
        image_numpy = image.numpy()
        if self.keep_aspect_ratio:
            image_numpy = self.resize_maintaining_aspect_ratio(image.numpy(), self.width, self.height,
                                                               self.padding_color)
        else:
            image_numpy = cv2.resize(image_numpy, (self.width, self.height))

        image.update(image_numpy)

        return image, label
