import numpy as np
import math
import random
import os

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.constants import ALPHA, P
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz


class AdaptivePhases(AdaptiveLiteSense):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 min_skip: int,
                 collect_mode: CollectMode,
                 max_collected: Optional[int] = None,
                 max_window_size: int = 0,
                 epsilon: int = 0.2
                 ):
        super().__init__(collection_rate=collection_rate,
                         threshold=threshold,
                         num_seq=num_seq,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=max_skip,
                         min_skip=min_skip,
                         collect_mode=collect_mode,
                         max_collected=max_collected,
                         max_window_size=max_window_size)
        # Policy parameters
        self._seq_length = seq_length
        self._deviations = []
        self._means = []

        self.mean = 0
        self.dev = 0

        # Default parameters
        self._collection_rate = 0.9

        # Load collection rates for phases
        base = os.path.dirname(__file__)
        collection_rates_path = os.path.join(base, '../saved_models', 'epilepsy', 'collection_rates.json.gz')
        self._collection_rates = read_json_gz(collection_rates_path)[str(collection_rate)]
        self._label_idx = 0

        # Fit default uniform policy
        target_samples = int(self._collection_rate * seq_length)

        skip = max(1.0 / self._collection_rate, 1)
        frac_part = skip - math.floor(skip)

        self._skip_indices: List[int] = []

        index = 0
        while index < seq_length:
            self._skip_indices.append(index)

            if (target_samples - len(self._skip_indices)) == (seq_length - index - 1):
                index += 1
            else:
                r = self._rand.uniform()
                if r > frac_part:
                    index += int(math.floor(skip))
                else:
                    index += int(math.ceil(skip))

        self._skip_indices = self._skip_indices[:target_samples]
        self._skip_idx = 0

        self._window_size = 0
        self._max_window_size = max_window_size

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_PHASES
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def update(self, collection_ratio: float, seq_idx: int):
        pass

    def collect(self, measurement: np.ndarray):
        measurement = (measurement[0]**2 + measurement[1]**2 + measurement[2]**2)**0.5

        self.mean = (1.0 - self._alpha) * self.mean + self._alpha * measurement
        self.dev = (1.0 - self._beta) * self.dev + self._beta * np.abs(self.mean - measurement)

        self._deviations.append(self.dev)
        self._means.append(self.mean)


    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self, label):
        collection_rate = self._collection_rates[self._label_idx]
        self._label_idx += 1

        # Change uniform collection rate
        target_samples = int(collection_rate * self._seq_length)

        skip = max(1.0 / collection_rate, 1)
        frac_part = skip - math.floor(skip)

        self._skip_indices: List[int] = []

        index = 0
        while index < self._seq_length:
            self._skip_indices.append(index)

            if (target_samples - len(self._skip_indices)) == (self._seq_length - index - 1):
                index += 1
            else:
                r = self._rand.uniform()
                if r > frac_part:
                    index += int(math.floor(skip))
                else:
                    index += int(math.ceil(skip))

        self._skip_indices = self._skip_indices[:target_samples]
        self._skip_idx = 0
