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
from gambler.utils.constants import ALPHA, P
from gambler.utils.controller import Controller
from gambler.utils.action import Action
from gambler.utils.distribution import load_distribution
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data


class AdaptiveTraining(AdaptiveLiteSense):

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
        self._distribution = load_distribution('moving_dist.txt')
        self._interp = np.interp(self._distribution, [min(self._distribution), max(self._distribution)], [0, 100])
        self._deviations = []
        self._means = []

        self._window = 100

        self._total_samples = seq_length*num_seq
        self._total_time = self._total_samples
        self._budget = self._total_samples*collection_rate
        self._samples_collected = 0

        self.mean = 0
        self.dev = 0

        # Default parameters
        self._collection_rate = collection_rate # start at uniform collection rate

        # Load collection rates for phases
        base = os.path.dirname(__file__)
        collection_rates_path = os.path.join(base, '../saved_models', 'epilepsy', 'collection_rates.json.gz')
        self._collection_rates = read_json_gz(collection_rates_path)

        # Fit LinearRegression model
        # X = []
        # y = []
        # for key, value in self._collection_rates.items():
        #     for i, cr in enumerate(value):
        #         X.append([float(key), i])
        #         y.append(cr)

        # Load validation set 
        validation_rates_path = os.path.join(base, '../saved_models', 'epilepsy', 'collection_rates_validation.json.gz')
        self._collection_rates = read_json_gz(validation_rates_path)
        X_val = []
        y_val = []
        for key, value in self._collection_rates.items():
            for i, cr in enumerate(value):
                X_val.append([float(key), i])
                y_val.append(cr)

        # Load training data
        # training_set = pd.read_csv('train.csv')
        # X = training_set.iloc[:, 0:2].values.tolist()
        # y = training_set.iloc[:, 2:].values.ravel()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)
        # self.model = RandomForestRegressor(max_depth=8, random_state=0).fit(X_train, y_train)
        self.model = pickle.load(open('saved_models/rf', 'rb'))
        # pickle.dump(self.model, open('saved_models/rf', 'wb'))

        self._label_idx = -1

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
        return PolicyType.ADAPTIVE_TRAINING
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def update(self, collection_ratio: float, seq_idx: int):
        # Update estimate mean reward            
        _dev = sum(self._deviations)/len(self._deviations)
        _mean = sum(self._means)/len(self._means)
        cov = (_dev/_mean)/0.18

        # Update time
        self._total_samples = self._total_samples - self._window
        leftover = self._budget - self._samples_collected
        
        budget_left = leftover / self._total_samples

        time_left = self._total_samples
        time_spent = (self._total_time - self._total_samples)
        alpha = time_left / self._total_time
        beta = time_spent / self._total_time

        global_cr = self.model.predict(np.array([cov, self._collection_rate]).reshape(1, -1))[0]
        local_cr = self.model.predict(np.array([cov, budget_left]).reshape(1, -1))[0]

        collection_rate = (alpha * global_cr) + (beta * local_cr)

        # Change uniform collection rate
        if self._skip_idx < len(self._skip_indices):
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

        self._samples_collected += 1


    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self, label):
        # print("====================================LABEL CHANGE================================================")
        self._label_idx = label
