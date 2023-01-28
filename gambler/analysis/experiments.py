#!/bin/python3

import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
import os
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools
import random
import statistics
from collections import Counter

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--policies', type=str, nargs='+')
    parser.add_argument('--randomize', type=str, default='expected')
    parser.add_argument('--rand-amount', type=int, default=1)
    parser.add_argument('--num-runs', type=int, default=100)
    parser.add_argument('--should-overwrite', action='store_true')
    parser.add_argument('--window-size', type=int, default=20)
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    policies = [
        'uniform',
        'adaptive_heuristic',
        'adaptive_deviation',
        'adaptive_uniform',
        'adaptive_gambler',
    ]

    window_sizes = {
        'epilepsy': 10,
        'uci_har': 50, 
        'wisdm': 20,
        'trajectories': 50,
        'pedestrian': 20, 
        'temperature': 40,
        'pavement': 30, 
        'haptics': 5,
        'eog': 50
    }

    NUM_RUNS = len(budgets) if args.randomize == 'expected' else args.num_runs

    if args.policies:
        policies = args.policies

    output_file = os.path.join('results', f'{args.randomize}_distribution_{args.rand_amount}.json.gz')
    results_dict = read_json_gz(output_file) if os.path.exists(output_file) else dict()

    # Build dictionary to hold results
    for dataset in args.datasets:
        if dataset not in results_dict:
            results_dict[dataset] = dict()
        for policy in policies:
            results_dict[dataset][policy] = []

    for dataset in args.datasets:
        # Load dataset
        fold = 'train'
        true_inputs, true_labels = load_data(dataset_name=dataset, fold=fold)
        unique_labels = list(set(true_labels))
        num_seq = len(true_inputs)
        num_labels = len(unique_labels)

        fold = 'test'
        true_inputs, true_labels = load_data(dataset_name=dataset, fold=fold)        

        # Run experiment for each policy
        for i in range(NUM_RUNS):
            if args.randomize == 'skewed':
                # Select random budget
                budget = random.choice(budgets)
            else:
                # Select collection rate
                budget = budgets[i]

            # Generate random distribution
            classes = []
            if args.randomize == 'skewed':
                seq_left = num_seq
                while seq_left > 0:
                    rand_label = unique_labels[random.randrange(num_labels)]
                    rand_length = max(1, random.randint(0, seq_left//args.rand_amount))
                    seq_left -= rand_length
                    classes += [rand_label for _ in range(rand_length)]

            # Rearrange data for experiments
            inputs, labels = get_data(classes, true_inputs, true_labels)

            # Unpack the shape
            num_seqs, seq_lengths, num_features = inputs.shape

            # Merge sequences into continous stream
            inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

            # Unpack the shape
            seq_length, num_features = inputs.shape

            for policy_name in policies:
                # Make the policy
                collect_mode = 'tiny'
                window_size = window_sizes[dataset] if policy_name == 'adaptive_gambler' else 20
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

                max_num_seq = num_seqs

                policy.init_for_experiment(num_sequences=max_num_seq)

                # Run the policy
                policy_result = run_policy(policy=policy,
                                        sequence=inputs,
                                        should_enforce_budget=True)

                collected_list = policy_result.measurements
                collected_indices = flatten(policy_result.collected_indices)

                # Stack collected features into a numpy array
                collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]

                # Reconstruct the sequence
                reconstructed = reconstruct_sequence(measurements=collected,
                                                    collected_indices=collected_indices,
                                                    seq_length=seq_length)

                # Calculate error
                mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)

                # Store results
                results_dict[dataset][policy_name].append([budget, mae])

    # Save results
    save_json_gz(results_dict, output_file)
