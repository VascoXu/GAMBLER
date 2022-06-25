import numpy as np

import os.path
from collections import deque, OrderedDict
from typing import Tuple, List, Dict, Any, Optional

from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy


def run_policy(policy: BudgetWrappedPolicy, sequence: np.ndarray, window: tuple, seq_num: int, should_enforce_budget: bool) -> PolicyResult:
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

    # Unpack collection information
    collected_within_window, curr_window_size = window

    # Lists to hold the results
    collected_list: List[np.ndarray] = []
    collected_indices: List[int] = []

    # List to hold
    collection_ratios: List[float] = []

    # Execute the policy on the given sequence
    for seq_idx in range(seq_length):
        should_collect = policy.should_collect(seq_idx=seq_idx, seq_num=seq_num)

        # if should_enforce_budget:
            # should_collect = should_collect and not policy.has_exhausted_budget()
    
        # if should_collect and not policy.has_exhausted_budget():
        if should_collect:
            measurement = sequence[seq_idx]
            policy.collect(measurement=measurement)

            collected_list.append(measurement.reshape(1, -1))
            collected_indices.append(seq_idx)

            collected_within_window += 1

        curr_window_size += 1

        if curr_window_size == policy.max_window_size:
            collection_ratio = (collected_within_window/policy.max_window_size)
            collection_ratios.append(collection_ratio)

            # Update threshold based on control theory
            if policy.policy_type == PolicyType.ADAPTIVE_CONTROLLER or policy.policy_type == PolicyType.ADAPTIVE_GREEDY or policy.policy_type == PolicyType.ADAPTIVE_BANDIT or policy.policy_type == PolicyType.ADAPTIVE_SIGMA or policy.policy_type == PolicyType.BANDIT_BUDGET or policy.policy_type == PolicyType.SIGMA_BUDGET:
                policy.update(collection_ratio, seq_idx)

            # Reset window parameters
            curr_window_size = 0
            collected_within_window = 0

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list) if len(collected_list) > 0 else []  # [K, D]
    
    return PolicyResult(measurements=collected,
                        collected_indices=collected_indices,
                        collection_ratios=collection_ratios,
                        num_collected=len(collected_indices),
                        collected_within_window=collected_within_window,
                        curr_window_size=curr_window_size)
