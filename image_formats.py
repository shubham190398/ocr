import os
import typing
import cv2
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

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