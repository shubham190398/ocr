import cv2
import typing
import numpy as np
import logging

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

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label


