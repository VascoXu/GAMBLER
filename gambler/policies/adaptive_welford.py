import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense


class AdaptiveWelford(AdaptiveLiteSense):

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

        self.mean = 0
        self.count = 0
        self.M2 = 0


    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_WELFORD


# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate


# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)


    def collect(self, measurement: np.ndarray):

        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        norm = np.sum(self._dev)

        if norm > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), self.min_skip)
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._estimate = measurement
        self._sample_skip = 0