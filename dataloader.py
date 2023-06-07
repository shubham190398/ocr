import os
import copy
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from augmentors import Augmentor
from transformers import Transformer
import logging


class DataLoader:
    def __init__(
            self,
            dataset: typing.Union[str, list, pd.DataFrame],
            data_preprocessors: typing.List[typing.Callable] = None,
            batch_size: int = 4,
            shuffle: bool = True,
            initial_epoch: int = 1,
            augmentors: typing.List[Augmentor] = None,
            transformers: typing.List[Transformer] = None,
            skip_validation: bool = True,
            limit: int = None,
            use_cache: bool = False,
            log_level: int = logging.INFO,
    ) -> None:
        self.dataset = dataset
        self.data_preprocessors = [] if data_preprocessors is None else data_preprocessors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._epoch = initial_epoch
        self.augmentors = [] if augmentors is None else augmentors
        self.transformers = [] if transformers is None else transformers
        self.skip_validation = skip_validation
        self.limit = limit
        self.use_cache = use_cache
        self._step = 0
        self.cache = {}
        self.on_epoch_end_remove = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not skip_validation:
            self.dataset = self.validate(dataset)
        else:
            self.logger.info("Skipping Dataset Validation")

        if limit:
            self.logger.info(f"Limiting dataset to {limit} samples")
            self.dataset = self.dataset[:limit]

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    @property
    def augmentors(self) -> typing.List[Augmentor]:
        return self.augmentors

    @augmentors.setter
    def augmentors(self, augmentors: typing.List[Augmentor]):
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                if self.augmentors is not None:
                    self.augmentors.append(augmentor)
                else:
                    self.augmentors = [augmentor]

            else:
                self.logger.warning(f"Augmentor {augmentor} is not an instance of Augmentor.")

    @property
    def transformers(self) -> typing.List[Transformer]:
        return self.transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self.transformers is not None:
                    self.transformers.append(transformer)
                else:
                    self.transformers = [transformer]

            else:
                self.logger.warning(f"Transformer {transformer} is not an instance of Transformer.")

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def on_epoch_end(self):
        self._epoch += 1

        if self.shuffle:
            np.random.shuffle(self.dataset)

        for remove in self.on_epoch_end_remove:
            self.logger.warning(f"Removing {remove} from dataset")
            self.dataset.remove(remove)

        self.on_epoch_end_remove = []

