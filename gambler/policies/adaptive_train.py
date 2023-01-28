import numpy as np
import math
import random

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution


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
                 window_size: int = 0):
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
                         window_size=window_size)
        
        # Default parameters
        self._seq_length = seq_length
        self._dataset = dataset
        
        # Policy parameters
        self._average_dev: List[float] = []
        self._training_data: List[List[float]] = []
    
    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_TRAIN

    @property
    def training_data(self):
        return self._training_data


    def update(self, collection_ratio: float, seq_idx: int, window: tuple):
        # Ensure budget was not exhausted
        if len(self._average_dev) == 0:
            return
          
        dev = sum(self._average_dev)/len(self._average_dev)

        # Hold training data
        self._training_data.append([np.sum(dev), self._collection_rate, collection_ratio])
        
        # Reset parameters
        self._average_dev: List[float] = []


    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        # Update policy parameters
        self._average_dev.append(self._dev)

        norm = np.sum(self._dev)
        if norm > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), self.min_skip)
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._estimate = measurement
        self._sample_skip = 0


    def reset(self):
        super().reset()
        self._training_data: List[List[float]] = []