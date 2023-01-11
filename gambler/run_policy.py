from gc import collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import h5py
import random
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import List, Tuple
from collections import Counter

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


def write_dataset(inputs, labels, filename, output_folder):
    # Write new dataset to h5 file
    partition_folder = os.path.join('./datasets', output_folder)
    if not os.path.exists(partition_folder):
        os.mkdir(partition_folder)

    partition_inputs = inputs
    partition_output = labels
    with h5py.File(os.path.join(partition_folder, f'{filename}.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
        input_dataset.write_direct(partition_inputs)

        output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
        output_dataset.write_direct(partition_output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=[])
    parser.add_argument('--fold', default='test')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--reconstruction-plot', type=str, default='window')
    parser.add_argument('--print-label-errors', action='store_true')
    parser.add_argument('--print-label-rates', action='store_true')
    parser.add_argument('--graph-cr', action='store_true')
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--max-skip', type=int, default=0)
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)
    
    # Load the data
    input_seqs, labels = load_data(dataset_name=args.dataset, fold=args.fold)
    labels = labels.reshape(-1)

    # Rearrange data for experiments
    inputs, labels = get_data(args.classes, input_seqs, labels)

    # Unpack the shape
    # num_seqs, seq_lengths, num_features = input_seqs.shape
    num_seqs, seq_lengths, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

    # Unpack the shape
    seq_length, num_features = inputs.shape

    # Make the policy
    collect_mode = 'tiny'
    policy = BudgetWrappedPolicy(name=args.policy,
                                 num_seq=num_seqs,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 dataset=args.dataset,
                                 collection_rate=args.collection_rate,
                                 collect_mode=collect_mode,
                                 window_size=args.window_size,
                                 model=args.model,
                                 max_skip=args.max_skip)


    errors: List[float] = []
    measurements: List[np.ndarray] = []
    estimate_list: List[np.ndarray] = []
    
    collected_counts = defaultdict(list)
    collected: List[List[int]] = []
    
    collected_nums: List[int] = []
    collection_ratios: List[List[float]] = []
    window_labels: List[int] = []

    max_num_seq = num_seqs if args.max_num_samples is None else min(num_seqs, args.max_num_samples)

    policy.init_for_experiment(num_sequences=max_num_seq)

    # Run the policy
    policy_result = run_policy(policy=policy,
                                sequence=inputs,
                                should_enforce_budget=True)

    collected_list = policy_result.measurements
    estimate_list = policy_result.estimate_list
    collected_indices = flatten(policy_result.collected_indices)
    errors = policy_result.errors

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]

    # Reconstruct the sequence
    reconstructed = reconstruct_sequence(measurements=collected,
                                        collected_indices=collected_indices,
                                        seq_length=seq_length)

    # Calculate errors
    mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)
    norm_mae = normalized_mae(y_true=inputs, y_pred=reconstructed)
    rmse = mean_squared_error(y_true=inputs, y_pred=reconstructed, squared=False)
    norm_rmse = normalized_rmse(y_true=inputs, y_pred=reconstructed)
    r2 = r2_score(y_true=inputs, y_pred=reconstructed, multioutput='variance_weighted')

    num_measurements = seq_length
    num_collected = len(collected_indices)

    # Compute collection rate for each label (class)
    if args.print_label_rates:
        unique_labels = set(labels)
        rates_dict = {}

        left, right = 0, 0
        first, last = 0, 0
        for label in unique_labels:
            # Find sequences of each label
            label_indices = np.where(labels == label)[0]
            last += len(label_indices)*seq_lengths

            try:
                right = next(i for i,val in enumerate(collected_indices) if val > last) - 1
            except:
                right = len(collected_indices) - 1

            rates_dict[label] = (right-left+1)/len(collected_indices)

            left = right+1
            first = last+1
        
        print(rates_dict)

    # Compute error for each label (class)
    if args.print_label_errors:
        unique_labels = set(labels)
        errors_dict = {}

        for label in unique_labels:
            # Find sequences of each label
            label_indices = np.where(labels == label)[0]
            first = label_indices[0]
            last = label_indices[len(label_indices)-1]

            # Compute error of true signal and reconstructed
            y_seq, y_length, _ = input_seqs[first:last].shape
            y_true = input_seqs[first:last].reshape(y_seq*y_length, num_features)
            errors_dict[label] = mean_absolute_error(y_true=y_true, y_pred=reconstructed[first*seq_lengths:last*seq_lengths])

        print(errors_dict)

    # Graph collection rate for each window
    if args.graph_cr:
        with plt.style.context('seaborn-ticks'):
            plt.plot(policy_result.collection_ratios, label='Collection Rate')
            plt.title(f'{args.policy.upper()} ({args.dataset.upper()}): Collection Rate over Time')
            plt.ylim([0, 1])
            plt.legend()
            plt.show()

    print('{0:.5f},{1},{2} ({3})'.format(mae, num_collected, num_measurements, num_collected/num_measurements))

    if args.reconstruction_plot == 'window':
        # Plot window reconstruction
        data_idx = np.argmax(errors)
        left = data_idx*args.window_size
        right = left+args.window_size

        true_inputs = inputs[left:right]
        reconstructed = reconstructed[left:right]

        estimates = reconstructed
        collected_idx = policy_result.window_indices[data_idx]

    elif args.reconstruction_plot == 'sequence':
        # Plot sequence reconstruction
        data_idx = 2
        left = data_idx*num_seqs
        right = left+seq_lengths

        true_inputs = inputs[left:right]
        reconstructed = reconstructed[left:right]

        estimates = reconstructed
        collected_idx = collected_indices[left:right]

    elif args.reconstruction_plot == 'all':
        # Plot reconstruction for all data
        true_inputs = inputs
        estimates = reconstructed
        collected_idx = collected_indices

    print('Max Error: {0:.5f} (Idx: {1})'.format(errors[data_idx], data_idx))
    print('Max Error Collected: {0}'.format(len(collected_idx)))

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots()

        # xs = list(range(args.window_size))
        xs = list(range(len(reconstructed)))
        ax1.plot(xs, true_inputs[:, args.feature], label='True', color='royalblue')
        ax1.plot(xs, reconstructed[:, args.feature], label='Inferred', color='orange')
        # ax1.scatter(collected_idx, estimates[collected_idx, args.feature], marker='o', color='orange')

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Feature Value')

        ax1.legend()

        plt.show()
