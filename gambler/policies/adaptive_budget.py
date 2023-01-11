import numpy as np
import math

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense


class AdaptiveBudget(AdaptiveLiteSense):

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
                         model=model,
                         collect_mode=collect_mode,
                         max_collected=max_collected,
                         window_size=window_size)
        # Default parameters
        self._num_seq = num_seq
        self._seq_length = seq_length
        self._window_idx = 0

        # Policy parameters
        self._budget = int(seq_length*collection_rate)
        self._total_samples = seq_length
        self._total_time = self._total_samples
        self._budget = int(self._total_samples*collection_rate)
        self._budget_cutoff = self._total_time * 0.5
        self._samples_collected = 0
        self._samples_seen = 0
        self._skip_indices: List[int] = []
        self._adaptive = True

        self._prev_total_samples = self._total_samples
        self._prev_budget = self._budget


    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_BUDGET

        
    def should_collect(self, seq_idx: int, window: tuple) -> bool:
        (window_idx, window_num) = window
        self._samples_seen += 1

        if self._adaptive:
            # Execute Adaptive Policy
            if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
                return True

            self._sample_skip += 1
            return False
        else:
            # Execute Uniform Policy
            if (self._skip_idx < len(self._skip_indices) and self._window_idx == self._skip_indices[self._skip_idx]):
                self._skip_idx += 1
                self._window_idx += 1
                return True

            self._window_idx += 1

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

        self._samples_collected += 1


    def update(self, collection_ratio: float, seq_idx: int, window: tuple):
        self._total_samples -= self._window_size
        leftover = self._budget - self._samples_collected
        
        budget_left = leftover / self._total_samples if self._total_samples > 0 else 0

        self._prev_budget = self._samples_collected
        self._prev_total_samples = leftover

        if self._samples_collected >= self._budget_cutoff and self._adaptive:
            self._count = 0
            self._adaptive = False

        if not self._adaptive:
            self._count += 1

            if budget_left >= 1:
                collection_rate = 1.0
            
            collection_rate = budget_left

            # Fit default uniform policy
            target_samples = int(collection_rate * self._window_size)

            skip = 1 if collection_rate == 0 else max(1.0 / collection_rate, 1)
            frac_part = skip - math.floor(skip)

            self._skip_indices: List[int] = []

            index = 0
            while index < self._window_size:
                self._skip_indices.append(index)
    
                if (target_samples - len(self._skip_indices)) == (self._window_size - index - 1):
                    index += 1
                else:
                    index += int(math.ceil(skip))

            self._skip_idx = 0
            self._window_idx = 0
            self._skip_indices = self._skip_indices[:target_samples]


    def reset(self):
        super().reset()
        self._skip_idx = 0