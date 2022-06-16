import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_policy import AdaptivePolicy


class AdaptiveHeuristic(AdaptivePolicy):

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_HEURISTIC

    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (self._sample_skip > 0):
            self._sample_skip -= 1
            return False

        return True

    def collect(self, measurement: np.ndarray):
        if len(measurement.shape) >= 2:
            measurement = measurement.reshape(-1)

        diff = np.sum(np.abs(self._estimate - measurement))
        self._estimate = measurement

        if diff >= self.threshold:
            self._current_skip = self.min_skip
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._sample_skip = self._current_skip