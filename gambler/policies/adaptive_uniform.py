import numpy as np
import math

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense


class AdaptiveUniform(AdaptiveLiteSense):

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
        
        self._budget = (seq_length*num_seq)*collection_rate
        self._sample_count = 0
        self._num_seq = num_seq
        self._seq_length = seq_length
        self._budget_cutoff = self._budget * 0.8
        self._window_size = 0
        self._max_window_size = max_window_size
        self._skip_indices: List[int] = []

        self._adaptive = True

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_UNIFORM
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if self._sample_count >= self._budget_cutoff and self._adaptive and seq_idx == 0:
            leftover = self._budget - self._sample_count
            target_samples = int(leftover/(self._num_seq-seq_num))
            leftover_rate = target_samples / self._seq_length

            skip = max(1.0 / leftover_rate, 1)
            frac_part = skip - math.floor(skip)

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

            self._adaptive = False

        if self._adaptive:
            # Run Adaptive Policy
            if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
                return True

            self._sample_skip += 1
            return False
        else:
            # Run Uniform Policy
            if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
                self._skip_idx += 1
                return True

            return False

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

        self._sample_count += 1

    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self, label):
        pass    