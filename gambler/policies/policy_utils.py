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


def run_policy(policy: BudgetWrappedPolicy, sequence: np.ndarray, should_enforce_budget: bool) -> PolicyResult:
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
    curr_window_size = 0
    collected_within_window = 0
    window_num = 0
    sample_num = 0

    # Lists to hold the results
    # collected_idx: List[List[int]] = []
    collected_idx: List[int] = []
    measurement_list: List[np.ndarray] = []
    collected_list: List[np.ndarray] = []
    collected_indices: List[int] = []

    # List to hold
    collection_ratios: List[float] = []
    errors: List[float] = []

    # Execute the policy on the given sequence
    for seq_idx in range(seq_length):
        should_collect = policy.should_collect(seq_idx=seq_idx, window=(curr_window_size, window_num)) #TODO: change to window NUM

        if should_collect and not policy.has_exhausted_budget():
            measurement = sequence[seq_idx]

            policy.collect(measurement=measurement)

            collected_list.append(measurement.reshape(1, -1))
            measurement_list.append(measurement.reshape(1, -1))

            collected_idx.append(seq_idx)
            collected_indices.append(sample_num)

            collected_within_window += 1

        curr_window_size += 1
        sample_num += 1
        
        # Update policies after each window
        if curr_window_size == policy.window_size:
            collection_ratio = (collected_within_window/policy.window_size)
            collection_ratios.append(collection_ratio)

            # Update policy 
            if policy.policy_type == PolicyType.ADAPTIVE_GAMBLER or policy.policy_type == PolicyType.ADAPTIVE_UNIFORM or policy.policy_type == PolicyType.ADAPTIVE_TRAIN or policy.policy_type == PolicyType.ADAPTIVE_BUDGET or policy.policy_type == PolicyType.ADAPTIVE_BANDIT:
                policy.update(collection_ratio, seq_idx, window=(curr_window_size, window_num))

            policy.step(seq_idx=0, count=collected_within_window)

            # Reconstruct the sequence
            if collected_within_window == 0:
                reconstructed = np.zeros([policy.window_size, num_features])
            else:
                # Stack collected features into a numpy array
                collected = np.vstack(measurement_list) # [K, D]
                reconstructed = reconstruct_sequence(measurements=collected,
                                                    collected_indices=collected_indices,
                                                    seq_length=policy.window_size)

            left = window_num*policy.window_size
            right = left+policy.window_size
            true = sequence[left:right]
            error = mean_absolute_error(y_true=true, y_pred=reconstructed)

            if collected_within_window > 0:
                pass
                # collected_seq = idx + 1
            else:
                error = 1

            # Record the policy results
            errors.append(error)
            measurement_list.append(reconstructed)

            # Reset window parameters
            curr_window_size = 0
            collected_within_window = 0
            sample_num = 0
            window_num += 1

            # Reset measurements parameters
            measurement_list: List[np.ndarray] = []
            collected_indices: List[int] = []

    # reconstructed = np.vstack([np.expand_dims(r, axis=0) for r in measurement_list])  # [N, T, D]
    # reconstructed = reconstructed.reshape(-1, num_features)

    # Handle remaining/leftover measurements
    # remaining_samples = seq_length-reconstructed.shape[0]
    # collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([remaining_samples, num_features])  # [K, D]
    # if len(collected_list) > 0:
    #     leftover_reconstructed = reconstruct_sequence(measurements=collected,
    #                                         collected_indices=collected_indices,
    #                                         seq_length=remaining_samples)
    # else:
    #     leftover_reconstructed = collected
    # reconstructed = np.concatenate((reconstructed, leftover_reconstructed))

    training_data = policy.training if policy.policy_type == PolicyType.ADAPTIVE_TRAIN else []

    return PolicyResult(measurements=collected_list,
                        collected_indices=collected_idx,
                        collection_ratios=collection_ratios,
                        num_collected=len(collected_idx),
                        training_data=training_data)
