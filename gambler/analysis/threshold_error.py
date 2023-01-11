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
from gambler.utils.analysis import normalized_mae, normalized_rmse, mean_absolute_percentage_error
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.analysis import normalized_rmse
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH, MARKER, MARKER_SIZE


WIDTH = 0.15

DATASET = 'epilepsy'
BUDGET = 0.7
THRESHOLD = 0.15 # for budget of 70%
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
        # mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)
        mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)

        # Store results
        results_dict[policy_name]['reconstructed'] = reconstructed
        results_dict[policy_name]['indices'] = policy_result.collected_indices
        results_dict[policy_name]['mae'] = mae
        results_dict[policy_name]['errors'] = errors

    return results_dict


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

    # Walking+Running
    walk_run = np.concatenate((inputs[210:220], inputs[210:220], inputs[330:350]))
    res_walk_run = execute_policy(walk_run)

    # Sequence One (Energy Waste)
    seq_one = np.concatenate((inputs[210:225], inputs[210:225], inputs[210:220]))
    res_one = execute_policy(seq_one)
    seq_one = merge_sequence(seq_one)

    # Sequence Two (Early Exhaustion)
    seq_two = inputs[333:373]
    random.shuffle(seq_two)
    res_two = execute_policy(seq_two)
    seq_two = merge_sequence(seq_two)

    errors = dict()
    errors['uniform'] = [res_walk_run['uniform']['mae'], res_one['uniform']['mae'], res_two['uniform']['mae']]
    errors['adaptive'] = [res_walk_run['adaptive_heuristic']['mae'], res_one['adaptive_heuristic']['mae'], res_two['adaptive_heuristic']['mae']]

    # Normalize the errors
    for i, error in enumerate(errors['adaptive']):
        errors['adaptive'][i] = errors['adaptive'][i]/errors['uniform'][i]

    for i, error in enumerate(errors['uniform']):
        errors['uniform'][i] = 1

    xs = np.arange(0, 3)
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        bar_plot(ax, errors, total_width=.8, single_width=1, colors=list(COLORS.values()), legend=['Uniform Sampling', 'Adaptive Sampling'])
        ax.set_xticks(xs)
        ax.set_xticklabels(['Walking + Running', 'Walking', 'Running'])
        ax.set_xlabel('Events', fontsize=14)
        ax.set_ylabel('MAE (Normalized to Uniform)', fontsize=14)
        plt.title(f'Limitations of Threshold-Based Adaptive Sampling', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()        

        