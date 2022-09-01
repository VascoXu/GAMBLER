import numpy as np
import math
import random

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.constants import ALPHA, P
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution
from gambler.utils.measurements import translate_measurement


class AdaptiveTrain(AdaptiveLiteSense):

    def __init__(self,
                 collection_rate: float,
                 dataset: str,
                 threshold: float,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 min_skip: int,
                 collect_mode: CollectMode,
                 model: str,
                 max_collected: Optional[int] = None,
                 max_window_size: int = 0,
                 epsilon: int = 0.2
                 ):
        super().__init__(collection_rate=collection_rate,
                         dataset=dataset,
                         threshold=threshold,
                         num_seq=num_seq,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=max_skip,
                         min_skip=min_skip,
                         collect_mode=collect_mode,
                         model=model,
                         max_collected=max_collected,
                         max_window_size=max_window_size)
        
        # Default parameters
        self._seq_length = seq_length
        self._dataset = dataset
        
        # Policy parameters
        self._moving_dev = 0
        self._moving_mean = 0
        self._dev_count = 0
        self._mean_count = 0

        self._training = []
        self._count = 0
    
        self.mean = 0
        self.dev = 0

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_TRAIN

    @property
    def training(self):
        return self._training

    def update(self, collection_ratio: float, seq_idx: int):
        # Check if budget was reached
        if self._dev_count == 0:
            return

        # Update estimate mean reward            
        _dev = self._moving_dev / self._dev_count
        _mean = self._moving_mean / self._mean_count
        _cov = (_dev/_mean)

        # Write to training data
        self._training.append([_cov, self._collection_rate, collection_ratio])
        
        # Reset parameters
        self._moving_dev = 0
        self._dev_count = 0

    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        # Generating training set
        measurement = translate_measurement(measurement, self._dataset)
        self.mean = (1.0 - self._alpha) * self.mean + self._alpha * measurement
        self.dev = (1.0 - self._beta) * self.dev + self._beta * np.abs(self.mean - measurement)

        # Update policy parameters
        self._moving_dev += self.dev
        self._moving_mean += self.mean
        self._dev_count += 1
        self._mean_count += 1

        # Should collect
        norm = np.sum(self._dev)

        if norm > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), self.min_skip)
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._estimate = measurement
        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._training = []