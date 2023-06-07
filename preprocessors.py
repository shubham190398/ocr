import os
import typing
import numpy as np
import logging
from image_formats import NormalImage


class ImageReader:
    def __init__(
            self,
            image_class,
            log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class
