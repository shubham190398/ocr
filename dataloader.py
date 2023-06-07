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
        self.epoch = initial_epoch
        self.augmentors = [] if augmentors is None else augmentors
        self.transformers = [] if transformers is None else transformers
        self.skip_validation = skip_validation
        self.limit = limit
        self.use_cache = use_cache
        self.cache = {}
        self.on_epoch_and_remove = []
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
