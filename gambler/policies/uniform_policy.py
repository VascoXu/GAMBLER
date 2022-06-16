import numpy as np
import math

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.policy import Policy


class UniformPolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 collect_mode: CollectMode,
                 max_window_size: int = 0):
        super().__init__(collection_rate=collection_rate,
                         num_seq=num_seq,
                         num_features=num_features,
                         seq_length=seq_length,
                         collect_mode=collect_mode,
                         )

        self._max_window_size = max_window_size

        target_samples = int(collection_rate * seq_length)

        skip = max(1.0 / collection_rate, 1)
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

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.UNIFORM

    @property
    def max_window_size(self) -> int:
        return self._max_window_size

    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self):
        pass