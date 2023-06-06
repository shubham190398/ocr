import os
import typing
import cv2
from abc import ABC, abstractmethod
import numpy as np


class NormalImage(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def center(self) -> tuple:
        pass

    @abstractmethod
    def RGB(self) -> np.ndarray:
        pass

    @abstractmethod
    def HSV(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, image: np.ndarray):
        pass

    @abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class CVImage(NormalImage):
    init_successful = False

    def __init__(
            self,
            image: typing.Union[str, np.ndarray],
            method: int = cv2.IMREAD_COLOR,
            path: str = "",
            color: str = "BGR"
    ) -> None:
        super().__init__()

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self.image = cv2.imread(image, method)
            self.path = image
            self.color = "BGR"

        elif isinstance(image, np.ndarray):
            self.image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(f"Image must be either path to image or numpy.ndarray, not {type(image)}")

        self.method = method

        if self.image is None:
            return None

        self.init_successful = True
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]

    @property
    def image(self) -> np.ndarray:
        return self.image

    @image.setter
    def image(self, value: np.ndarray):
        self.image = value

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self.image
        elif self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self.image = image
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
            self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]
            return self

        else:
            raise TypeError(f"image must be numpy.ndarray, not {type(image)}")

    def flip(self, axis: int = 0):
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")
        self.image = self.image[:, ::-1] if axis == 0 else self.image[::-1]
        return self

    def numpy(self) -> np.ndarray:
        return self.image

    def __call__(self) -> np.ndarray:
        return self.image
