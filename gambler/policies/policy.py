import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode


class Policy:

    def __init__(self,
                 collection_rate: float,
                 num_seq: int,
                 num_features: int,
                 seq_length: int,
                 collect_mode: CollectMode,
                 ):

        
        self._estimate = np.zeros((num_features, ))  # [D]
        self._num_seq = num_seq
        self._num_features = num_features
        self._seq_length = seq_length
        self._collection_rate = collection_rate
        self._collect_mode = collect_mode

        self._rand = np.random.RandomState(seed=78362)

        # Track the average number of measurements sent
        self._measurement_count = 0
        self._seq_count = 0

    @property
    def seq_length(self) -> int:
        return self._seq_length

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def collection_rate(self) -> float:
        return self._collection_rate

    @property
    def collect_mode(self) -> CollectMode:
        return self._collect_mode

    def collect(self, measurement: np.ndarray):
        self._estimate = np.copy(measurement.reshape(-1))  # [D]

    def reset(self):
        self._estimate = np.zeros((self._num_features, ))  # [D]

    def step(self, count: int, seq_idx: int):
        self._measurement_count += count
        self._seq_count += 1

    def __str__(self) -> str:
        return '{0}-{1}'.format(self.policy_type.name.lower(), self.collect_mode.name.lower())

    def as_dict(self) -> Dict[str, Any]:
        return {
            'policy_name': self.policy_type.name,
            'collection_rate': self.collection_rate,
            'collect_mode': self.collect_mode.name,
        }

    @property
    def policy_type(self) -> PolicyType:
        raise NotImplementedError()

    def should_collect(self, seq_idx: int, window: tuple) -> bool:
        raise NotImplementedError()