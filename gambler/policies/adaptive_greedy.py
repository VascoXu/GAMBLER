import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.constants import ALPHA, P
from gambler.utils.controller import Controller


class AdaptiveGreedy(AdaptiveLiteSense):

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
                 epsilon: int = 0.99
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


        self._epsilon = epsilon
        self._gamma = 0.99

        self._window_size = 0
        self._max_window_size = max_window_size

        self._controller = Controller(alpha=ALPHA, P=P, setpoint=collection_rate)

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_GREEDY
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
            return True

        self._sample_skip += 1
        return False

    def update(self, collection_ratio: float, seq_idx: int): 
        # Epsilon-Greedy Approach
        rand = self._rand.random()
        if rand < self._epsilon: 
            # explore (trust adaptive)
            self._epsilon *= self._gamma
        else: 
            # exploit (trust controller)
            updated_threshold = self._controller.step(collection_ratio, self.threshold)
            self.set_threshold(updated_threshold)        

    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        norm = np.sum(self._dev)

        if norm > self.threshold:
            self._current_skip = max(int(self._current_skip / 2), self.min_skip)
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._estimate = measurement
        self._sample_skip = 0