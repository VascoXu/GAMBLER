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


class SigmaBudget(AdaptiveLiteSense):

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
                 epsilon: int = 0.2
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
        # Policy parameters
        self._seq_length = seq_length
        self._total_samples = seq_length*num_seq
        self._budget = (seq_length*num_seq)*collection_rate
        self._num_seq = num_seq
        self._distribution = load_distribution('moving_dist.txt')
        self._interp = np.interp(self._distribution, [min(self._distribution), max(self._distribution)], [0, 100])
        
        self._deviations = []
        self._means = []

        self.mean = 0
        self.dev = 0

        self._offset = 0.3
        self._decay = 0.9
        
        self._sample_count = 0
        self._switched = False
        self._count = 0

        # Default parameters
        self._collection_rate = 0.9

        # Fit default uniform policy
        target_samples = int(self._collection_rate * seq_length)

        skip = max(1.0 / self._collection_rate, 1)
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

        self._window_size = 0
        self._max_window_size = max_window_size

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.SIGMA_BUDGET
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
      # Reached halfway point
        if self._count == (self._total_samples/2) and not self._switched:
            # Determine leftover budget
            leftover = round(self._budget - self._sample_count)
            target_samples = int(leftover/(self._num_seq-seq_num))
            leftover_rate = target_samples / self._seq_length

            # Switch to uniform for remaining  of the budget
            old_idx = self._skip_indices[self._skip_idx]

            skip = max(1.0 / leftover_rate, 1)
            frac_part = skip - math.floor(skip)

            self._skip_indices: List[int] = []

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
            
            self._skip_idx = len(self._skip_indices)-1
            for (i, val) in enumerate(self._skip_indices):
                if val >= old_idx:
                    self._skip_idx = i
                    break

            self._switched = True

        self._count += 1

        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def update(self, collection_ratio: float, seq_idx: int):
        # Do not update on Uniform
        if self._switched == True:
            return

        # Check if budget was reached
        if len(self._deviations) == 0:
            return

        # Calculate percentile of observed deviation
        dev = np.interp(sum(self._dev), [min(self._distribution), max(self._distribution)], [0, 100])
        percentile = np.percentile(self._interp, dev)

        # Update estimate mean reward            
        _dev = sum(self._deviations)/len(self._deviations)
        _mean = sum(self._means)/len(self._means)
        cov = (_dev/_mean)/0.18
        collection_rate = round(min(cov, 1), 1)
        # collection_rate = round(round(min(cov, 1), 1) + self._offset, 1)
        self._offset *= self._decay

        # Update collection rate
        if (self._skip_idx < len(self._skip_indices) and self._collection_rate != collection_rate):
            old_idx = self._skip_indices[self._skip_idx]
            target_samples = int(collection_rate * self._seq_length)

            skip = max(1.0 / collection_rate, 1)
            frac_part = skip - math.floor(skip)

            self._skip_indices: List[int] = []

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

            self._skip_idx = len(self._skip_indices)-1
            for (i, val) in enumerate(self._skip_indices):
                if val >= old_idx:
                    self._skip_idx = i
                    break
            
        self._collection_rate = collection_rate
        self._deviations = []

    def collect(self, measurement: np.ndarray):
        measurement = (measurement[0]**2 + measurement[1]**2 + measurement[2]**2)**0.5

        # self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        # self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)
        self.mean = (1.0 - self._alpha) * self.mean + self._alpha * measurement
        self.dev = (1.0 - self._beta) * self.dev + self._beta * np.abs(self.mean - measurement)

        # self._deviations.append(sum(self._dev))
        self._deviations.append(self.dev)
        self._means.append(self.mean)

        self._sample_count += 1


    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self, label):
        pass
        # print("====================================LABEL CHANGE================================================")