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

DATASET = 'phone_accel'
THRESHOLD = 0.15 # for budget of 70%
WINDOW_SIZE = 20
SEQ_LENGTH = 40

POLICIES = ['uniform', 'adaptive_deviation', 'adaptive_uniform', 'adaptive_gambler']
COLLECTION_RATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def merge_sequence(sequence):
    # Merge feature dimensions
    merged = np.zeros((sequence.shape[0], 1))
    for i, sample in enumerate(sequence):
        x, y, z = sample
        merged[i] = ((x**2 + y**2 + z**2)**0.5)
    return merged


def execute_policy(inputs):
    # Unpack the shape
    num_seqs, seq_lengths, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

    # Default parameters
    results_dict = dict()

    for policy_name in POLICIES:
        results_dict[policy_name] = dict()

    # Unpack the shape
    num_seqs = 1
    seq_length, num_features = inputs.shape

    for policy_name in POLICIES:
        mean_error = 0
        for collection_rate in COLLECTION_RATES:
            # Make the policy
            collect_mode = 'tiny'
            policy = BudgetWrappedPolicy(name=policy_name,
                                        num_seq=num_seqs,
                                        seq_length=seq_length,
                                        num_features=num_features,
                                        dataset=DATASET,
                                        collection_rate=collection_rate,
                                        collect_mode=collect_mode,
                                        window_size=WINDOW_SIZE,
                                        model='',
                                        max_skip=0)

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
            mean_error += mae

        # Store results
        results_dict[policy_name]['mae'] = mean_error/len(COLLECTION_RATES)

    return results_dict



if __name__ == '__main__':
    # Seed for reproducible results
    random.seed(42)

    errors = dict()
    for policy in POLICIES:
        errors[policy] = []

    # Load SIMILAR dataset
    inputs, labels = load_data(dataset_name='phone_accel', fold='test')
    similar_res = execute_policy(inputs)
    for policy in POLICIES:
        errors[policy].append(similar_res[policy]['mae'])

    # Load the PINGPONG dataset
    inputs, labels = load_data(dataset_name='pingpong', fold='test')
    cooking_res = execute_policy(inputs)
    for policy in POLICIES:
        errors[policy].append(cooking_res[policy]['mae'])

    # Load the COOKING dataset
    inputs, labels = load_data(dataset_name='cooking', fold='test')
    cooking_res = execute_policy(inputs)
    for policy in POLICIES:
        errors[policy].append(cooking_res[policy]['mae'])


    # Load the DRIVING dataset
    inputs, labels = load_data(dataset_name='driving', fold='test')
    driving_res = execute_policy(inputs)
    for policy in POLICIES:
        errors[policy].append(driving_res[policy]['mae'])
  
    # Compute mean of datasets
    for policy in POLICIES:
        errors[policy].append(statistics.mean(errors[policy]))        

    # Normalize the errors
    for policy in reversed(POLICIES):
        for i, error in enumerate(errors[policy]):
            errors[policy][i] = errors[policy][i]/errors['uniform'][i]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        legend = ['Uniform', 'Adaptive Deviation', 'Adaptive Uniform', 'Adaptive GAMBLER']
        xticks = ['Walking+Jogging\nSitting+Standing', 'Ping Pong', 'Cooking', 'Driving', 'Overall']

        xs = bar_plot(ax, errors, total_width=.8, single_width=1, colors=list(COLORS.values()), legend=legend)

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)

        ax.set_xticks(np.arange(0, len(xticks)))
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Events', fontsize=14)
        ax.set_ylabel('MAE (Normalized to Uniform)', fontsize=14)
        plt.title(f'Adaptive Sampling on Mobile Phone Accelerometer', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()        

        