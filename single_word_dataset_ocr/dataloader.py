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
        self._augmentors = [] if augmentors is None else augmentors
        self._transformers = [] if transformers is None else transformers
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
        return self._augmentors

    @augmentors.setter
    def augmentors(self, augmentors: typing.List[Augmentor]):
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                if self._augmentors is not None:
                    self._augmentors.append(augmentor)
                else:
                    self._augmentors = [augmentor]

            else:
                self.logger.warning(f"Augmentor {augmentor} is not an instance of Augmentor.")

    @property
    def transformers(self) -> typing.List[Transformer]:
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]

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

    @staticmethod
    def validate_list_dataset(dataset: list) -> list:
        validated_data = [data for data in tqdm(dataset, desc="Validating Dataset") if os.path.exists(data[0])]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset.")

        return validated_data

    # noinspection PyMethodMayBeStatic
    def validate(self, dataset: typing.Union[str, list, pd.DataFrame]) -> typing.Union[list, str]:
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return DataLoader.validate_list_dataset(dataset)
        elif isinstance(dataset, pd.DataFrame):
            # noinspection PyTypeChecker
            return DataLoader.validate_list_dataset(dataset.values.tolist())
        else:
            raise TypeError("Dataset must be a path, list or pandas dataframe.")

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[typing.Any, typing.Any]:
        if shuffle:
            np.random.shuffle(self.dataset)

        train_data_provider, val_data_provider = copy.deepcopy(self), copy.deepcopy(self)
        train_data_provider.dataset = self.dataset[:int(len(self.dataset) * split)]
        val_data_provider.dataset = self.dataset[int(len(self.dataset) * split):]

        return train_data_provider, val_data_provider

    def to_csv(self, path: str, index: bool = False) -> None:
        df = pd.DataFrame(self.dataset)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        self._step = index
        start_index = index * self.batch_size

        batch_indexes = [i for i in range(start_index, start_index + self.batch_size) if i < len(self.dataset)]
        batch_annotations = [self.dataset[index] for index in batch_indexes]

        return batch_annotations

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item

    def process_data(self, batch_data):
        if self.use_cache and batch_data[0] in self.cache:
            data, annotation = copy.deepcopy(self.cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self.data_preprocessors:
                data, annotation = preprocessor(data, annotation)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self.on_epoch_end_remove.append(batch_data)
                return None, None

            if self.use_cache and batch_data[0] not in self.cache:
                self.cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        for objects in [self._augmentors, self._transformers]:
            for _object in objects:
                data, annotation = _object(data, annotation)

        if not isinstance(data, np.ndarray):
            data = data.numpy()

        if not isinstance(annotation, (np.ndarray, int, float, str, np.uint8, float)):
            annotation = annotation.numpy()

        return data, annotation

    def __getitem__(self, index: int):
        dataset_batch = self.get_batch_annotations(index)

        batch_data, batch_annotations = [], []
        for index, batch in enumerate(dataset_batch):
            data, annotation = self.process_data(batch)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue

            batch_data.append(data)
            batch_annotations.append(annotation)

        return np.array(batch_data), np.array(batch_annotations)
