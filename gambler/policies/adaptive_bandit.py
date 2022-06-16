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


class AdaptiveBandit(AdaptiveLiteSense):

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
        self._mean_actions = []

        # Epsilon-greedy parameters
        self._epsilon = 1
        self._sigma = 0.7
        self._step_size = 0.2
        self._decay = 0.9
        self._initial = 10

        # Bandit parameters
        self._actions = []
        for i in range(2, 10):
            self._actions.append(Action(i*0.1))
        self._j = len(self._actions) - 1

        self._delta = 1/len(self._actions)

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
        return PolicyType.ADAPTIVE_BANDIT
        
    def should_collect(self, seq_idx: int, seq_num: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def update(self, collection_ratio: float, seq_idx: int):
        # Calculate percentile of observed deviation
        dev = np.interp(sum(self._dev), [min(self._distribution), max(self._distribution)], [0, 100])
        percentile = np.percentile(self._interp, dev)

        # Update estimate mean reward
        # reward = 1/abs(percentile/100-self._collection_rate)
        reward = 1/abs(sum(self._dev)-self._collection_rate)
        q_estimate = self._actions[self._j].reward
        q_a = q_estimate + self._step_size*(reward - q_estimate)
        self._actions[self._j].reward = q_a

        print('Eps: ', self._epsilon)

        # print(round(self._actions[self._j].val, 2))

        # print(f'ACTION ({round(self._actions[self._j].val, 2)}):')
        # print('====================================================')
        # print(f'percentile: {percentile} | dev: {sum(self._dev)}')
        # print(f'prev estimate: {q_estimate}')
        # print(f'received reward: {reward}')
        # print(f'updted estimate: {self._actions[self._j].reward}')
        # print('====================================================')

        if self._initial < len(self._actions):
            # Try all actions
            collection_rate = self._actions[self._initial].val
            self._j = self._initial
            self._initial += 1        
        else:
            # Epsilon-Greedy Approach
            rand = self._rand.random()
            if rand < self._epsilon: 
                # explore
                idx = random.choice(range(len(self._actions)))
                collection_rate = round(self._actions[idx].val,2)
                # print(f'EXPLORE: {round(collection_rate, 2)}')
                self._j = idx
            else: 
                # exploit
                self._j = np.argmax([action.reward for i,action in enumerate(self._actions)])
                collection_rate = round(self._actions[self._j].val,2)
                # print(f'EXPLOIT: {round(collection_rate, 2)} REWARD: {self._actions[self._j].reward}')

            # update epsilon
            TD_error = abs(reward - q_estimate)
            top = (1-math.e**((-(self._step_size)*TD_error)/self._sigma))
            bottom = (1+math.e**((-(self._step_size)*TD_error)/self._sigma))

            f = top / bottom 
            # print('Previous Eps: ', self._epsilon)
            self._epsilon = self._delta * f + (1-self._delta) * self._epsilon
            # print('After Eps: ', self._epsilon)                


        # Update collection rate
        if (self._skip_idx < len(self._skip_indices) and self._collection_rate != collection_rate):
            # print("================AFTER================")
            # print(self._skip_idx, seq_idx, len(self._skip_indices), self._skip_indices[self._skip_idx])
            # for (i, item) in enumerate(self._skip_indices):
            #     print((i, item), end=" ,")
            # print()

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

            # print("================AFTER================")
            # print(self._skip_idx, seq_idx, len(self._skip_indices))
            # for (i, item) in enumerate(self._skip_indices):
            #     print((i, item), end=" ,")
            # print()
            
        self._collection_rate = collection_rate

    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)
        self._deviations.append(sum(self._dev))

    def reset(self):
        super().reset()
        self._skip_idx = 0

    def reset_params(self):
        print("====================================LABEL CHANGE================================================")
        for i in range(len(self._actions)):
            pass
            # print(f'Action {round((i+2)*0.1, 2)}: {self._actions[i].reward}')

        # self._epsilon = 0.2