import numpy as np
import math
import random

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.policy import Policy


class UniformPolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 dataset: str,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 collect_mode: CollectMode,
                 window_size: int = 0):
        super().__init__(collection_rate=collection_rate,
                         num_seq=num_seq,
                         num_features=num_features,
                         seq_length=seq_length,
                         collect_mode=collect_mode,
                         )
        # =======================
        # For tracking deviation
        self._alpha = 0.7
        self._beta = 0.7

        self._mean = np.zeros(shape=(num_features, ))  # [D]
        self._dev = np.zeros(shape=(num_features, ))

        self._dataset = dataset
        self._window_size = window_size
        self._window_idx = 0

        self._average_dev: List[float] = []
        self._training_data: List[List[float]] = []
        # =======================

        if num_seq == 1:
            target_samples = int(collection_rate * window_size)
            iter_size = window_size
        else:
            target_samples = int(collection_rate * seq_length)
            iter_size = seq_length

        skip = max(1.0 / collection_rate, 1)
        frac_part = skip - math.floor(skip)

        self._skip_indices: List[int] = []

        index = 0
        while index < iter_size:
            self._skip_indices.append(index)
            if (target_samples - len(self._skip_indices)) == (iter_size - index - 1):
                index += 1
            else:
                r = self._rand.uniform()
                if r > frac_part:
                    index += int(math.floor(skip))
                else:
                    index += int(math.ceil(skip))    

        self._skip_indices = self._skip_indices[:target_samples]
        self._skip_idx = 0

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.UNIFORM


    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def training_data(self):
        return self._training_data

    @property
    def deviation(self):
        return np.sum(self._dev)


    def update(self, collection_ratio: float, seq_idx: int, window: tuple, measurements: List[np.ndarray]):
        self._window_idx = 0
        self._skip_idx = 0

        # Ensure budget was not exhausted
        if len(self._average_dev) == 0:
            return

        dev = sum(self._average_dev)/len(self._average_dev)

        # Hold training data
        self._training_data.append([np.sum(self._dev), self._collection_rate, collection_ratio])
        
        # Reset parameters
        self._average_dev: List[float] = []


    def should_collect(self, seq_idx: int, window: tuple) -> bool:
        self._window_idx += 1
        if (self._skip_idx < len(self._skip_indices) and self._window_idx-1 == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False
        

    def collect(self, measurement: np.ndarray):
        # Track deviation for experiments
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        # Update policy parameters
        self._average_dev.append(self._dev)


    def reset(self):
        super().reset()
        self._skip_idx = 0
        self._training_data: List[List[float]] = []