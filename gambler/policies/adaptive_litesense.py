
import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_policy import AdaptivePolicy


class AdaptiveLiteSense(AdaptivePolicy):

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
                 epsilon: int = 0.6,
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
        self._alpha = 0.7
        self._beta = 0.7

        self._mean = np.zeros(shape=(num_features, ))  # [D]
        self._dev = np.zeros(shape=(num_features, ))

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_LITESENSE

    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
            return True

        self._sample_skip += 1
        return False

    def collect(self, measurement: np.ndarray):
        if len(measurement.shape) >= 2:
            measurement = measurement.reshape(-1)  # [D]

        updated_mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        updated_dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(updated_mean - measurement)

        diff = np.sum(updated_dev - self._dev)

        if diff >= self.threshold:
            self._current_skip = max(self._current_skip - 1, 0)
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._estimate = measurement

        self._mean = updated_mean
        self._dev = updated_dev

        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._mean = np.zeros(shape=(self.num_features, ))  # [D]
        self._dev = np.zeros(shape=(self.num_features, ))  # [D]

    def reset_params(self, label):
        pass