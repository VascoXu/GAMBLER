from re import S
import numpy as np
import pandas as pd
import math
import random
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data


class AdaptiveGambler(AdaptiveLiteSense):

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
                 window_size: int = 0,
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
                         window_size=window_size)
        
        # Default parameters
        self._seq_length = seq_length
        self._dataset = dataset
        self._window_size = window_size
        self._window_idx = 0

        # Policy parameters
        self._avg_dev: List[float] = []
        self._avg_mean: List[float] = []
        self.mean = 0
        self.dev = 0

        # Budget parameters
        self._samples_collected = 0
        self._total_samples = seq_length
        self._total_time = self._total_samples
        self._budget = self._total_samples*collection_rate

        # Model parameters
        self._collection_rate = collection_rate
        model_name = f'saved_models/{dataset}/models/gambler_model_{self._window_size}' if model == '' else model
        self.model = pickle.load(open(model_name, 'rb'))

        # Fit default uniform policy
        target_samples = int(collection_rate * window_size)

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


    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_GAMBLER


    def should_collect(self, seq_idx: int, window: tuple) -> bool:
        self._window_idx += 1

        if (self._skip_idx < len(self._skip_indices) and (self._window_idx-1) == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1

            return True

        return False


    def update(self, collection_ratio: float, seq_idx: int, window: tuple):
        # Update deviation
        _dev = 0 if not self._avg_dev else sum(self._avg_dev)/len(self._avg_dev)
        _mean = 0 if not self._avg_mean else sum(self._avg_mean)/len(self._avg_mean)

        # Update time
        self._total_samples -= self._window_size
        leftover = self._budget - self._samples_collected
        budget_left = leftover / self._total_samples if self._total_samples > 0 else 0

        """
        time_left = self._total_samples
        time_spent = (self._total_time - self._total_samples)
        
        # Weighted average using time remaining
        alpha = time_left / self._total_time
        beta = time_spent / self._total_time
        """

        # Weighted average using budget remaining
        alpha = leftover / self._budget
        beta = self._samples_collected / self._budget

        # Determine collection rates
        global_cr = self.model.predict(np.array([np.sum(_dev), self._collection_rate]).reshape(1, -1))[0]
        local_cr = self.model.predict(np.array([np.sum(_dev), budget_left]).reshape(1, -1))[0]
        collection_rate = (alpha * global_cr) + (beta * budget_left)

        # Set collection rate to 1 when there is surplus in budget 
        if budget_left >= 1:
            collection_rate = 1.0

        # Fit default uniform policy
        target_samples = int(collection_rate * self._window_size)
        skip = 1 if collection_rate == 0 else max(1.0 / collection_rate, 1)
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
    
        # Reset parameters
        self._avg_dev = []
        self._avg_mean = []


    def collect(self, measurement: np.ndarray):
        self.mean = (1.0 - self._alpha) * self.mean + self._alpha * measurement
        self.dev = (1.0 - self._beta) * self.dev + self._beta * np.abs(self.mean - measurement)
        
        # Update policy parameters
        self._avg_dev.append(self.dev)
        self._avg_mean.append(self.mean)

        self._samples_collected += 1


    def reset(self):
        super().reset()
        
        self._avg_dev: List[float] = []
        self._avg_mean: List[float] = []
        self.mean = 0
        self.dev = 0

        self._samples_collected = 0
        self._total_samples = self._seq_length        

        self._skip_idx = 0
        self._window_idx = 0