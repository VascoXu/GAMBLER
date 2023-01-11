import json
import os
import numpy as np
import random
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statistics import geometric_mean

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.analysis import normalized_rmse
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH


DATASET = 'epilepsy'
BUDGET = 0.7
THRESHOLD = 0.15 # for budget of 50%
WINDOW_SIZE = 20
SEQ_LENGTH = 40


def merge_sequence(sequence):
    # Merge feature dimensions
    merged = np.zeros((sequence.shape[0], 1))
    for i, sample in enumerate(sequence):
        x, y, z = sample
        merged[i] = ((x**2 + y**2 + z**2)**0.5)
    return merged


def execute_policy(inputs):
    # Default parameters
    policies = ['uniform', 'adaptive_heuristic']
    results_dict = dict()

    for policy_name in policies:
        results_dict[policy_name] = dict()

    # Unpack the shape
    num_seqs = 1
    seq_length, num_features = inputs.shape

    for policy_name in policies:
        # Make the policy
        collect_mode = 'tiny'
        policy = BudgetWrappedPolicy(name=policy_name,
                                    num_seq=num_seqs,
                                    seq_length=seq_length,
                                    num_features=num_features,
                                    dataset=DATASET,
                                    collection_rate=BUDGET,
                                    collect_mode=collect_mode,
                                    window_size=WINDOW_SIZE,
                                    model='',
                                    max_skip=0)
        
        if policy_name == 'adaptive_heuristic':
            policy.set_threshold(THRESHOLD)

        max_num_seq = num_seqs

        policy.init_for_experiment(num_sequences=max_num_seq)

        # Run the policy
        policy_result = run_policy(policy=policy,
                                sequence=inputs,
                                should_enforce_budget=True)

        collected_list = policy_result.measurements
        collected_indices = flatten(policy_result.collected_indices)
        errors = policy_result.errors

        # Stack collected features into a numpy array
        collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]

        # Reconstruct the sequence
        reconstructed = reconstruct_sequence(measurements=collected,
                                            collected_indices=collected_indices,
                                            seq_length=seq_length)

        # Calculate error
        mae = mean_squared_error(y_true=inputs, y_pred=reconstructed)

        # Store results
        results_dict[policy_name]['reconstructed'] = reconstructed
        results_dict[policy_name]['indices'] = policy_result.collected_indices
        # results_dict[policy_name]['indices'] = policy_result.window_indices
        results_dict[policy_name]['mae'] = mae
        results_dict[policy_name]['errors'] = errors
        print(f'Policy: {policy_name}, MAE: {mae}, Rate: {len(collected_indices)/seq_length}')

    # Plot window reconstruction
    data_idx = 0
    left = data_idx*WINDOW_SIZE
    right = left+WINDOW_SIZE

    # true_inputs = inputs[left:right]
    # reconstructed_uniform = results_dict['uniform']['reconstructed'][left:right]
    # reconstructed_adaptive = results_dict['adaptive_deviation']['reconstructed'][left:right]
    
    true_inputs = inputs
    reconstructed_uniform = results_dict['uniform']['reconstructed']
    reconstructed_adaptive = results_dict['adaptive_heuristic']['reconstructed']

    # Merge feature dimensions
    true_inputs = merge_sequence(true_inputs)
    reconstructed_uniform = merge_sequence(reconstructed_uniform)
    reconstructed_adaptive = merge_sequence(reconstructed_adaptive)

    collected_idx_uniform = flatten(results_dict['uniform']['indices'])
    collected_idx_adaptive = flatten(results_dict['adaptive_heuristic']['indices'])

    # collected_idx_uniform = results_dict['uniform']['indices'][data_idx]
    # collected_idx_adaptive = results_dict['adaptive_deviation']['indices'][data_idx]

    print('Uniform # Collected: ', len(collected_idx_uniform))
    print('Uniform Error: ', results_dict['uniform']['mae'])

    print('Adaptive # Collected: ', len(collected_idx_adaptive))
    print('Adaptive Error: ', results_dict['adaptive_heuristic']['mae'])

    # Build results dictionary
    res = dict()
    res['reconstructed_uniform'] = reconstructed_uniform
    res['reconstructed_adaptive'] = reconstructed_adaptive
    res['collected_idx_uniform'] = collected_idx_uniform
    res['collected_idx_adaptive'] = collected_idx_adaptive

    return res


if __name__ == '__main__':
    # Testing parameters
    classes = [1, 2]

    # Seed for reproducible results
    random.seed(42)

    # Run policy
    true_inputs, true_labels = load_data(dataset_name=DATASET, fold='test')

    # Rearrange data for experiments
    inputs, labels = get_data(classes, true_inputs, true_labels)

    # Unpack the shape
    num_seqs, seq_lengths, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

    # Sequence One (Energy Waste)
    seq_one = np.concatenate((inputs[210:225], inputs[210:225], inputs[210:220]))
    res_one = execute_policy(seq_one)
    seq_one = merge_sequence(seq_one)

    # Sequence Two (Early Exhaustion)
    seq_two = inputs[333:373]
    random.shuffle(seq_two)
    res_two = execute_policy(seq_two)
    seq_two = merge_sequence(seq_two)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(PLOT_SIZE[0] * 0.6, PLOT_SIZE[1]), sharex=True)

        # Plot first sequence
        xs = list(range(SEQ_LENGTH))
        ax1.set_ylim((0.6, 6))
        ax1.plot(xs, seq_one[:, 0], label='True', color='red', linewidth=5, alpha=0.65, zorder=1)
        ax1.plot(xs, res_one['reconstructed_uniform'][:, 0], label='Uniform', linestyle='dashed', dashes=(10, 5), color='royalblue')
        ax1.plot(xs, res_one['reconstructed_adaptive'][:, 0], label='Adaptive', linestyle='dashed', dashes=(20, 5), color='yellowgreen')
        ax1.scatter(res_one['collected_idx_uniform'], res_one['reconstructed_uniform'][res_one['collected_idx_uniform']], marker='o', color='royalblue', alpha=0.5, zorder=2)
        ax1.scatter(res_one['collected_idx_adaptive'], res_one['reconstructed_adaptive'][res_one['collected_idx_adaptive']], marker='o', color='yellowgreen', alpha=0.5, zorder=2)

        ax1.set_title('Energy Waste', fontsize=16)
        ax1.set_xlabel('Time Step', fontsize=14)
        ax1.set_ylabel('Total Acceleration', fontsize=14)

        ax1.legend()

        # Plot second sequence
        ax2.set_ylim((0.6, 6))
        ax2.plot(xs, seq_two[:, 0], label='True', color='red', linewidth=5, alpha=0.65, zorder=1)
        ax2.plot(xs, res_two['reconstructed_uniform'][:, 0], label='Uniform', linestyle='dashed', dashes=(10, 5), color='royalblue')
        ax2.plot(xs, res_two['reconstructed_adaptive'][:, 0], label='Adaptive', linestyle='dashed', dashes=(20, 5), color='yellowgreen')
        ax2.scatter(res_two['collected_idx_uniform'], res_two['reconstructed_uniform'][res_two['collected_idx_uniform']], marker='o', color='royalblue', alpha=0.5, zorder=2)
        ax2.scatter(res_two['collected_idx_adaptive'], res_two['reconstructed_adaptive'][res_two['collected_idx_adaptive']], marker='o', color='yellowgreen', alpha=0.5, zorder=2)

        ax2.set_title('Early Exhaustion', fontsize=16)
        ax2.set_xlabel('Time Step', fontsize=14)
        ax2.set_ylabel('Total Acceleration', fontsize=14)

        ax2.legend()

        fig.suptitle('Limitations of Threshold-Based Adaptive Sampling', fontsize=18)
        fig.tight_layout()
        plt.show()
