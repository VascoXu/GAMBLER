import numpy as np

import os.path
from collections import deque, OrderedDict
from typing import Tuple, List, Dict, Any, Optional
from gambler.policies.adaptive_budget import AdaptiveBudget

from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.policy import Policy
from gambler.policies.adaptive_deviation import AdaptiveDeviation
from gambler.policies.adaptive_litesense import AdaptiveLiteSense
from gambler.policies.adaptive_gambler import AdaptiveGambler
from gambler.policies.adaptive_train import AdaptiveTrain
from gambler.policies.adaptive_heuristic import AdaptiveHeuristic
from gambler.policies.adaptive_uniform import AdaptiveUniform
from gambler.policies.adaptive_budget import AdaptiveBudget
from gambler.policies.adaptive_bandit import AdaptiveBandit
from gambler.policies.uniform_policy import UniformPolicy


class BudgetWrappedPolicy(Policy):

    def __init__(self,
                 name: str,
                 num_seq: int,
                 seq_length: int,
                 num_features: int,
                 collect_mode: str,
                 collection_rate: float,
                 dataset: str,
                 **kwargs: Dict[str, Any]):
                 
        # Make the internal policy
        self._policy = make_policy(name=name,
                                   seq_length=seq_length,
                                   num_seq=num_seq,
                                   num_features=num_features,
                                   collect_mode=collect_mode,
                                   collection_rate=collection_rate,
                                   dataset=dataset,                                   
                                   **kwargs)

        # Call the base constructor to set the internal fields
        super().__init__(num_seq=num_seq,
                         seq_length=seq_length,
                         num_features=num_features,
                         collect_mode=self._policy.collect_mode,
                         collection_rate=collection_rate,)

        # Counters for tracking the energy consumption
        self._num_sequences: Optional[int] = None
        self._consumed_energy = 0.0
        self._num_collected = 0
        self._total_samples = 0
        self._budget = seq_length*collection_rate

        # Get the data distributions for possible random sequence generation
        # dirname = os.path.dirname(__file__)
        # distribution_path = os.path.join(dirname, '../datasets', dataset, 'distribution.json')
        # distribution = read_json(distribution_path)

        # self._data_mean = np.array(distribution['mean'])
        # self._data_std = np.array(distribution['std'])

    @property
    def policy_type(self) -> PolicyType:
        return self._policy.policy_type

    @property
    def window_size(self) -> int:
        return self._policy.window_size

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def threshold(self) -> float:
        return self._policy.threshold

    @property
    def training(self):
        return self._policy.training if self._policy.policy_type ==  PolicyType.ADAPTIVE_TRAIN else []

    def set_threshold(self, threshold: float):
        self._policy.set_threshold(threshold)

    def update(self, collection_ratio: float, seq_idx: int, window: tuple):
        self._policy.update(collection_ratio, seq_idx, window)
        
    def should_update(self):
        return self._policy.should_update()

    def should_collect(self, seq_idx: int, window: tuple) -> bool:
        self._total_samples += 1
        return self._policy.should_collect(seq_idx=seq_idx, window=window)

    def has_exhausted_budget(self) -> bool:
        return self._num_collected >= self._budget

    def collect(self, measurement: np.ndarray):
        self._policy.collect(measurement=measurement)
        self._num_collected += 1

    def reset(self):
        self._policy.reset()

    def init_for_experiment(self, num_sequences: int):
        self._num_sequences = num_sequences
        self._num_collected = 0

    def get_random_sequence(self) -> np.ndarray:
        rand_list: List[np.ndarray] = []

        for m, s in zip(self._data_mean, self._data_std):
            val = self._rand.normal(loc=m, scale=s, size=self.seq_length)  # [T]
            rand_list.append(np.expand_dims(val, axis=-1))

        return np.concatenate(rand_list, axis=-1)  # [T, D]

    def as_dict(self) -> Dict[str, Any]:
        result = super().as_dict()
        return result


def make_policy(name: str,
                num_seq: int,
                seq_length: int,
                num_features: int,
                collect_mode: str,
                collection_rate: float,
                dataset: str,
                **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    # Look up the data-specific precision and width
    base = os.path.dirname(__file__)
    quantize_path = os.path.join(base, '../datasets', dataset, 'quantize.json')

    quantize_dict = read_json(quantize_path)
    precision = quantize_dict['precision']
    width = quantize_dict['width']
    max_skip = quantize_dict.get('max_skip', 1)
    use_min_skip = quantize_dict.get('use_min_skip', False)
    threshold_factor = quantize_dict.get('threshold_factor', 1.0)

    if name == 'uniform':
        return UniformPolicy(collection_rate=collection_rate,
                             dataset=dataset,
                             num_seq=num_seq,
                             num_features=num_features,
                             seq_length=seq_length,
                             collect_mode=CollectMode[collect_mode.upper()],
                             window_size=kwargs['window_size'])
    elif name.startswith('adaptive'):
        # Look up the threshold path
        threshold_path = os.path.join(base, '../saved_models', dataset, 'thresholds_stream.json.gz')

        temp_name = name
        if name == 'adaptive_train' or name == 'adaptive_uniform' or name =='adaptive_elitesense' or name == 'adaptive_budget' or name =='adaptive_bandit':
            name = 'adaptive_deviation'

        did_find_threshold = False
        threshold_rate = collection_rate
        rate_str = str(round(threshold_rate, 2))

        if not os.path.exists(threshold_path):
            # print('WARNING: No threshold path exists.')
            threshold = 0.0
        else:
            thresholds = read_json_gz(threshold_path)
            if (name not in thresholds) or (collect_mode not in thresholds[name]) or (rate_str not in thresholds[name][collect_mode]):
                # print('WARNING: No threshold path exists.')
                threshold = 0.0
            else:
                threshold = thresholds[name][collect_mode][rate_str]
                did_find_threshold = True

        # Apply the optional data-specific threshold factor
        if isinstance(threshold_factor, OrderedDict):
            if rate_str in threshold_factor:
                threshold *= threshold_factor[rate_str]
        else:
            threshold *= threshold_factor

        # Get the optional max skip value
        if isinstance(max_skip, OrderedDict):
            max_skip_value = max_skip.get(rate_str, max_skip['default'])
        else:
            max_skip_value = max_skip

        max_skip_value += int(1.0 / threshold_rate)

        # Set the min skip
        if use_min_skip:
            if threshold_rate < 0.31:
                min_skip = 2
            elif threshold_rate < 0.51:
                min_skip = 1
            else:
                min_skip = 0
        else:
            min_skip = 0

        # Set the window size
        window_size = kwargs['window_size'] if 'window_size' in kwargs else 0

        # Reset name
        name = temp_name

        # For 'padded' policies, read the standard test log (if exists) to get the maximum number of collected values.
        # This is an impractical policy to use, as it requires prior knowledge of what the policy will do on the test
        # set. Nevertheless, we use this strategy to provide an 'ideal' baseline.
        max_collected = None

        if name == 'adaptive_heuristic':
            cls = AdaptiveHeuristic
        elif name == 'adaptive_litesense':
            cls = AdaptiveLiteSense
        elif name == 'adaptive_deviation':
            cls = AdaptiveDeviation
        elif name == 'adaptive_uniform':
            cls = AdaptiveUniform
        elif name =='adaptive_budget':
            cls = AdaptiveBudget
        elif name == 'adaptive_gambler':
            cls = AdaptiveGambler
        elif name == 'adaptive_bandit':
            cls = AdaptiveBandit
        elif name == 'adaptive_train':
            cls = AdaptiveTrain
        else:
            raise ValueError('Unknown adaptive policy with name: {0}'.format(name))

        return cls(collection_rate=collection_rate,
                   dataset=dataset,
                   threshold=threshold, 
                   num_seq=num_seq,
                   seq_length=seq_length,
                   num_features=num_features,
                   max_skip=max_skip_value,
                   min_skip=min_skip,
                   collect_mode=CollectMode[collect_mode.upper()],
                   model=kwargs['model'],
                   max_collected=max_collected,
                   window_size=window_size,
                   )
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))