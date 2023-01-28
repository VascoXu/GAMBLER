from curses import window
from gc import collect
import numpy as np

import os.path
from collections import deque, OrderedDict
from typing import Tuple, List, Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.utils.data_types import PolicyType, PolicyResult, CollectMode
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.misc_utils import flatten


def run_policy(policy: BudgetWrappedPolicy, sequence: np.ndarray, should_enforce_budget: bool, reconstruct: bool=False) -> PolicyResult:
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
    window_cnt = 0
    window_num = 0

    error = 1

    # Lists to hold the results
    measurement_list: List[np.ndarray] = []
    measurements: List[np.ndarray] = []
    estimate_list: List[np.ndarray] = []
    collected_indices: List[int] = []
    collected_idx: List[int] = []
    window_indices: List[List] = []
    window_idx: List[int] = []

    # List to hold
    collection_ratios: List[float] = []
    errors: List[float] = []

    # Execute the policy on the given sequence
    for seq_idx in range(seq_length):
        should_collect = policy.should_collect(seq_idx=seq_idx, window=(window_cnt, window_num))

        if should_collect and not (should_enforce_budget and policy.has_exhausted_budget()):
            measurement = sequence[seq_idx]

            policy.collect(measurement=measurement)

            measurements.append(measurement.reshape(1, -1))
            measurement_list.append(measurement.reshape(1, -1))

            collected_idx.append(seq_idx)
            window_idx.append(window_cnt)


        window_cnt += 1
        
        # Update policies after each window
        if (window_cnt == policy.window_size) or (seq_idx == seq_length-1):
            collected_within_window = len(window_idx)

            collection_ratio = (collected_within_window/window_cnt)
            collection_ratios.append(collection_ratio)

            # Update policy 
            if policy.policy_type != PolicyType.ADAPTIVE_DEVIATION and policy.policy_type != PolicyType.ADAPTIVE_HEURISTIC:
                policy.update(collection_ratio, seq_idx, window=(window_cnt, window_num))

            policy.step(seq_idx=0, count=collected_within_window)

            # Reconstruct the sequence
            if reconstruct:
                if collected_within_window == 0:
                    reconstructed = np.zeros([policy.window_size, num_features])
                else:
                    # Stack collected features into a numpy array
                    collected = np.vstack(measurement_list) # [K, D]
                    reconstructed = reconstruct_sequence(measurements=collected,
                                                        collected_indices=window_idx,
                                                        seq_length=window_cnt)
                    estimate_list.append(reconstructed)

                # Compute reconstruction error for window
                left = window_num*policy.window_size
                right = left+policy.window_size 
                true = sequence[left:right] if seq_idx < seq_length-1 else sequence[left:]

                if collected_within_window <= 0:
                    error = 1
                else:
                    error = mean_absolute_error(y_true=true, y_pred=reconstructed)

            # Record the policy results
            errors.append(error)
            window_indices.append(window_idx)
            collected_indices.append(collected_idx)

            # Reset window parameters
            window_cnt = 0
            window_num += 1

            # Reset measurements parameters
            measurement_list: List[np.ndarray] = []
            collected_idx: List[int] = []
            window_idx: List[int] = []


    return PolicyResult(measurements=measurements,
                        estimate_list=estimate_list,
                        window_indices=window_indices,
                        collected_indices=collected_indices,
                        collection_ratios=collection_ratios,
                        num_collected=len(flatten(collected_indices)),
                        errors=errors)
