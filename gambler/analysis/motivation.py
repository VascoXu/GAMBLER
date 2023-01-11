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


def merge_sequence(sequence):
    # Merge feature dimensions
    merged = np.zeros((sequence.shape[0], 1))
    for i, sample in enumerate(sequence):
        x, y, z = sample
        merged[i] = ((x**2 + y**2 + z**2)**0.5)
    return merged


if __name__ == '__main__':
    # Default parameters
    dataset = 'epilepsy'
    adaptive_policy = 'adaptive_heuristic'
    policies = ['uniform', adaptive_policy]
    results_dict = dict()

    for policy_name in policies:
        results_dict[policy_name] = dict()

    # Testing parameters
    budget = 0.7
    threshold = 0.15 # for budget of 70%
    classes = [1, 2]
    window_size = 20

    # Seed for reproducible results
    random.seed(42)

    # Run policy
    true_inputs, true_labels = load_data(dataset_name=dataset, fold='test')

    # Rearrange data for experiments
    inputs, labels = get_data(classes, true_inputs, true_labels)

    # Unpack the shape
    num_seqs, seq_lengths, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(num_seqs*seq_lengths, num_features)
    inputs = np.concatenate((inputs[210:220], inputs[210:220], inputs[330:350]))

    # Unpack the shape
    seq_length, num_features = inputs.shape

    for policy_name in policies:
        # Make the policy
        collect_mode = 'tiny'
        policy = BudgetWrappedPolicy(name=policy_name,
                                    num_seq=num_seqs,
                                    seq_length=seq_length,
                                    num_features=num_features,
                                    dataset=dataset,
                                    collection_rate=budget,
                                    collect_mode=collect_mode,
                                    window_size=window_size,
                                    model='',
                                    max_skip=0)
        
        if policy_name == adaptive_policy:
            policy.set_threshold(threshold)

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
        mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)
        # mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)

        # Store results
        results_dict[policy_name]['reconstructed'] = reconstructed
        results_dict[policy_name]['indices'] = policy_result.collected_indices
        results_dict[policy_name]['mae'] = mae
        results_dict[policy_name]['errors'] = errors

    # Plot window reconstruction
    data_idx = 0
    left = data_idx*window_size
    right = left+window_size

    # true_inputs = inputs[left:right]
    # reconstructed_uniform = results_dict['uniform']['reconstructed'][left:right]
    # reconstructed_adaptive = results_dict['adaptive_deviation']['reconstructed'][left:right]
    
    true_inputs = inputs
    reconstructed_uniform = results_dict['uniform']['reconstructed']
    reconstructed_adaptive = results_dict[adaptive_policy]['reconstructed']

    # Merge feature dimensions
    true_inputs = merge_sequence(true_inputs)
    reconstructed_uniform = merge_sequence(reconstructed_uniform)
    reconstructed_adaptive = merge_sequence(reconstructed_adaptive)

    estimates_uniform = reconstructed_uniform
    estimates_adaptive = reconstructed_adaptive

    collected_idx_uniform = flatten(results_dict['uniform']['indices'])
    collected_idx_adaptive = flatten(results_dict[adaptive_policy]['indices'])

    # collected_idx_uniform = results_dict['uniform']['indices'][data_idx]
    # collected_idx_adaptive = results_dict['adaptive_deviation']['indices'][data_idx]

    print('Uniform # Collected: ', len(collected_idx_uniform))
    print('Uniform Error: ', results_dict['uniform']['errors'])

    print('Adaptive # Collected: ', len(collected_idx_adaptive))
    print('Adaptive Error: ', results_dict[adaptive_policy]['errors'])

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots()
        ax1.set_ylim((0.6, 6))

        xs = list(range(len(true_inputs)))
        ax1.plot(xs, true_inputs[:, 0], label='True', color='red', linewidth=5, alpha=0.65, zorder=1)
        ax1.plot(xs, reconstructed_uniform[:, 0], label='Uniform', linestyle='dashed', dashes=(10, 5), color='royalblue')
        ax1.plot(xs, reconstructed_adaptive[:, 0], label='Adaptive', linestyle='dashed', dashes=(20, 5), color='yellowgreen')
        ax1.scatter(collected_idx_uniform, estimates_uniform[collected_idx_uniform], marker='o', color='royalblue', alpha=0.5, zorder=2)
        ax1.scatter(collected_idx_adaptive, estimates_adaptive[collected_idx_adaptive], marker='o', color='yellowgreen', alpha=0.5, zorder=2)

        ax1.set_title('Sampling on Accelerometer Data', fontsize=16)
        ax1.set_xlabel('Time Step', fontsize=14)
        ax1.set_ylabel('Total Acceleration', fontsize=14)

        ax1.legend(loc='upper left')

        plt.show()
