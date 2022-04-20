import numpy as np

import math
import random
import os.path
import time
from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional

from file_utils import read_json, read_pickle_gz, read_json_gz
from data_types import EncodingMode, EncryptionMode, PolicyType, PolicyResult, CollectMode


class Policy:

    def __init__(self,
                 collection_rate: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int,
                 collect_mode: CollectMode,
                 ):
        self._estimate = np.zeros((num_features, ))  # [D]
        self._precision = precision
        self._width = width
        self._num_features = num_features
        self._seq_length = seq_length
        self._collection_rate = collection_rate
        self._collect_mode = collect_mode

        self._rand = np.random.RandomState(seed=78362)

        # Track the average number of measurements sent
        self._measurement_count = 0
        self._seq_count = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def non_fractional(self) -> int:
        return self.width - self.precision

    @property
    def seq_length(self) -> int:
        return self._seq_length

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def collection_rate(self) -> float:
        return self._collection_rate

    @property
    def collect_mode(self) -> CollectMode:
        return self._collect_mode

    def collect(self, measurement: np.ndarray):
        self._estimate = np.copy(measurement.reshape(-1))  # [D]

    def reset(self):
        self._estimate = np.zeros((self._num_features, ))  # [D]

    def step(self, count: int, seq_idx: int):
        self._measurement_count += count
        self._seq_count += 1

    def __str__(self) -> str:
        return '{0}-{1}'.format(self.policy_type.name.lower(), self.collect_mode.name.lower())

    def as_dict(self) -> Dict[str, Any]:
        return {
            'policy_name': self.policy_type.name,
            'collection_rate': self.collection_rate,
            'width': self.width,
            'precision': self.precision,
            'collect_mode': self.collect_mode.name,
        }

    @property
    def policy_type(self) -> PolicyType:
        raise NotImplementedError()

    def should_collect(self, seq_idx: int) -> bool:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 min_skip: int,
                 max_skip: int,
                 collect_mode: CollectMode,
                 max_collected: Optional[int] = None):
        super().__init__(precision=precision,
                         width=width,
                         collection_rate=collection_rate,
                         num_features=num_features,
                         seq_length=seq_length,
                         collect_mode=collect_mode,)
        # Variables used to track the adaptive sampling policy
        self._max_skip = max_skip
        self._min_skip = min_skip

        assert self._max_skip >= self._min_skip, 'Must have a max skip > min_skip'

        self._current_skip = 0
        self._sample_skip = 0

        self._threshold = threshold

        self._max_collected = max_collected

    @property
    def max_skip(self) -> int:
        return self._max_skip

    @property
    def min_skip(self) -> int:
        return self._min_skip

    @property
    def max_num_groups(self) -> int:
        return self._max_num_groups

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def target_bytes(self) -> int:
        return self._target_bytes

    def set_threshold(self, threshold: float):
        self._threshold = threshold

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0


class AdaptiveHeuristic(AdaptivePolicy):

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_HEURISTIC

    def should_collect(self, seq_idx: int) -> bool:
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


class AdaptiveLiteSense(AdaptivePolicy):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 min_skip: int,
                 collect_mode: CollectMode,
                 max_collected: Optional[int] = None):
        super().__init__(collection_rate=collection_rate,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=max_skip,
                         min_skip=min_skip,
                         collect_mode=collect_mode,
                         max_collected=max_collected)
        self._alpha = 0.7
        self._beta = 0.7

        self._mean = np.zeros(shape=(num_features, ))  # [D]
        self._dev = np.zeros(shape=(num_features, ))

    def should_collect(self, seq_idx: int) -> bool:
        if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
            return True

        self._sample_skip += 1
        return False

    def collect(self, measurement: np.ndarray):
        if len(measurement.shape) >= 2:
            measurement = measurement.reshape(-1)  # [D]

        updated_mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        updated_dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(updated_mean - measurement)

        diff = np.sum(updated_dev - self._dev)

        if diff >= self.threshold:
            self._current_skip = max(self._current_skip - 1, 0)
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._estimate = measurement

        self._mean = updated_mean
        self._dev = updated_dev

        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._mean = np.zeros(shape=(self.num_features, ))  # [D]
        self._dev = np.zeros(shape=(self.num_features, ))  # [D]


class AdaptiveDeviation(AdaptiveLiteSense):

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_DEVIATION

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


class BudgetWrappedPolicy(Policy):

    def __init__(self,
                 name: str,
                 seq_length: int,
                 num_features: int,
                 collect_mode: str,
                 collection_rate: float,
                 dataset: str,
                 **kwargs: Dict[str, Any]):
                 
        # Make the internal policy
        self._policy = make_policy(name=name,
                                   seq_length=seq_length,
                                   num_features=num_features,
                                   collect_mode=collect_mode,
                                   collection_rate=collection_rate,
                                   dataset=dataset,
                                   **kwargs)

        # Call the base constructor to set the internal fields
        super().__init__(seq_length=seq_length,
                         num_features=num_features,
                         collect_mode=self._policy.collect_mode,
                         width=self._policy.width,
                         precision=self._policy.precision,
                         collection_rate=collection_rate,)

        # Counters for tracking the energy consumption
        self._num_sequences: Optional[int] = None

        # Get the data distributions for possible random sequence generation
        dirname = os.path.dirname(__file__)
        distribution_path = os.path.join(dirname, 'datasets', dataset, 'distribution.json')
        distribution = read_json(distribution_path)

        self._data_mean = np.array(distribution['mean'])
        self._data_std = np.array(distribution['std'])

    @property
    def policy_type(self) -> PolicyType:
        return self._policy.policy_type

    def set_threshold(self, threshold: float):
        self._policy.set_threshold(threshold)

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        return self._policy.encode(measurements=measurements,
                                   collected_indices=collected_indices)

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        return self._policy.decode(message=message)

    def should_collect(self, seq_idx: int) -> bool:
        return self._policy.should_collect(seq_idx=seq_idx)

    def collect(self, measurement: np.ndarray):
        self._policy.collect(measurement=measurement)

    def reset(self):
        self._policy.reset()

    def init_for_experiment(self, num_sequences: int):
        self._num_sequences = num_sequences

    def get_random_sequence(self) -> np.ndarray:
        rand_list: List[np.ndarray] = []

        for m, s in zip(self._data_mean, self._data_std):
            val = self._rand.normal(loc=m, scale=s, size=self.seq_length)  # [T]
            rand_list.append(np.expand_dims(val, axis=-1))

        return np.concatenate(rand_list, axis=-1)  # [T, D]

    def as_dict(self) -> Dict[str, Any]:
        result = super().as_dict()
        result['budget'] = self._budget
        result['energy_per_seq'] = self.energy_per_seq
        return result


class UniformPolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 collect_mode: CollectMode,
                 should_compress: bool):
        super().__init__(precision=precision,
                         width=width,
                         collection_rate=collection_rate,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         encoding_mode=EncodingMode.STANDARD,
                         collect_mode=collect_mode,
                         should_compress=should_compress)
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

    def should_collect(self, seq_idx: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def reset(self):
        super().reset()
        self._skip_idx = 0


def run_policy(policy: BudgetWrappedPolicy, sequence: np.ndarray, should_enforce_budget: bool, skipped_samples: int = 0) -> PolicyResult:
    """
    Executes the policy on the given sequence.
    Args:
        policy: The sampling policy
        sequence: A [T, D] array of features (D) for each element (T)
        should_enforce_budget: Whether to enforce the current energy budget
    Returns:
        A tuple of three elements:
            (1) A [K, D] array of the collected measurements
            (2) The K indices of the collected elements
            (3) The encoded message as a byte string
            (4) The energy required for this sequence
    """
    assert len(sequence.shape) == 2, 'Must provide a 2d sequence'

    # Reset all internal per-sequence counters
    policy.reset()

    # Unpack the shape
    seq_length, num_features = sequence.shape

    if should_enforce_budget and policy.has_exhausted_budget():
        rand_measurements = policy.get_random_sequence()
        return PolicyResult(measurements=rand_measurements,
                            collected_indices=list(range(seq_length)),
                            num_collected=seq_length,
                            energy=0.0,
                            num_bytes=0,
                            encoded=bytes())

    # Lists to hold the results
    collected_list: List[np.ndarray] = []
    collected_indices: List[int] = []
    collected_batches: List[np.ndarray] = []

    # Execute the policy on the given sequence
    skip_amount = 0
    skipped_samples = 5
    batch, batch_indices = [], []
    for seq_idx in range(seq_length):
        should_collect = policy.should_collect(seq_idx=seq_idx)

        should_skip = np.random.choice([0, 1], 1, p=[0.8, 0.2])[0]
        if should_skip and skipped_samples > 0:
            collected_batches.append(batch)
            batch, batch_indices = [], []
            skip_amount += skipped_samples
        
        if seq_idx + skip_amount >= seq_length:
            break

        if should_collect or should_skip:
            measurement = sequence[seq_idx + skip_amount]
            policy.collect(measurement=measurement)

            batch.append(measurement.reshape(1, -1))
            batch_indices.append(seq_idx + skip_amount)

            collected_list.append(measurement.reshape(1, -1))
            collected_indices.append(seq_idx + skip_amount)

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list)  # [K, D]

    return PolicyResult(measurements=collected,
                        collected_indices=collected_indices,
                        collected_batches=collected_batches,
                        batch_indices=batch_indices,
                        num_collected=len(collected_indices)
                        )


def make_policy(name: str,
                seq_length: int,
                num_features: int,
                collect_mode: str,
                collection_rate: float,
                dataset: str,
                **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    # Look up the data-specific precision and width
    base = os.path.dirname(__file__)
    quantize_path = os.path.join(base, 'datasets', dataset, 'quantize.json')

    quantize_dict = read_json(quantize_path)
    precision = quantize_dict['precision']
    width = quantize_dict['width']
    max_skip = quantize_dict.get('max_skip', 1)
    use_min_skip = quantize_dict.get('use_min_skip', False)
    threshold_factor = quantize_dict.get('threshold_factor', 1.0)

    if name == 'random':
        return RandomPolicy(collection_rate=collection_rate,
                            precision=precision,
                            width=width,
                            num_features=num_features,
                            seq_length=seq_length,
                            encryption_mode=EncryptionMode[encryption_mode.upper()],
                            collect_mode=CollectMode[collect_mode.upper()],
                            should_compress=should_compress)
    elif name == 'uniform':
        return UniformPolicy(collection_rate=collection_rate,
                             precision=precision,
                             width=width,
                             num_features=num_features,
                             seq_length=seq_length,
                             encryption_mode=EncryptionMode[encryption_mode.upper()],
                             collect_mode=CollectMode[collect_mode.upper()],
                             should_compress=should_compress)
    elif name.startswith('adaptive'):
        # Look up the threshold path
        threshold_path = os.path.join(base, 'saved_models', dataset, 'thresholds_block.json.gz')

        did_find_threshold = False
        threshold_rate = collection_rate
        rate_str = str(round(threshold_rate, 2))

        if not os.path.exists(threshold_path):
            print('WARNING: No threshold path exists.')
            threshold = 0.0
        else:
            thresholds = read_json_gz(threshold_path)
            if (name not in thresholds) or (collect_mode not in thresholds[name]) or (rate_str not in thresholds[name][collect_mode]):
                print('WARNING: No threshold path exists.')
                threshold = 0.0
            else:
                threshold = thresholds[name][collect_mode][rate_str]
                print("Threshold: ", threshold)
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
        else:
            raise ValueError('Unknown adaptive policy with name: {0}'.format(name))

        return cls(collection_rate=collection_rate,
                   threshold=threshold,
                   precision=precision,
                   width=width,
                   seq_length=seq_length,
                   num_features=num_features,
                   max_skip=max_skip_value,
                   min_skip=min_skip,
                   collect_mode=CollectMode[collect_mode.upper()],
                   max_collected=max_collected)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))