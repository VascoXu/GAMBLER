import numpy as np

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.policies.policy import Policy
from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode


class AdaptivePolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 dataset: str,
                 threshold: float,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 min_skip: int,
                 max_skip: int,
                 collect_mode: CollectMode,
                 model: str,
                 max_collected: Optional[int] = None,
                 max_window_size: int = 0,
                 epsilon:int = 0.6):
        super().__init__(collection_rate=collection_rate,
                         num_seq=num_seq,
                         num_features=num_features,
                         seq_length=seq_length,
                         collect_mode=collect_mode,)

        # Variables used to track the adaptive sampling policy
        self._max_skip = max_skip
        self._min_skip = min_skip

        assert self._max_skip >= self._min_skip, 'Must have a max skip > min_skip'

        self._current_skip = 0
        self._sample_skip = 0

        self._force_update = False

        self._threshold = threshold
        self._collection_rate = collection_rate

        self._max_collected = max_collected
        self._max_window_size = max_window_size

    @property
    def max_skip(self) -> int:
        return self._max_skip

    @property
    def min_skip(self) -> int:
        return self._min_skip

    @property
    def force_update(self) -> bool:
        return self._force_update

    @property
    def max_window_size(self) -> int:
        return self._max_window_size

    @property
    def threshold(self) -> float:
        return self._threshold
        
    def set_budget(self, budget: float):
        self._collection_rate = budget
        
    def set_update(self, update: bool):
        self._force_update = update

    def set_threshold(self, threshold: float):
        self._threshold = threshold

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0

    def reset_params(self, label):
        pass 