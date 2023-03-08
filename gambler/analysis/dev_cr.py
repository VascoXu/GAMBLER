import json
import os
import pandas as pd
import numpy as np
import random
import pickle
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
from gambler.utils.constants import DATASETS, WINDOW_SIZES
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH, MARKER, MARKER_SIZE


# DATASETS = ['epilepsy']
COLLECTION_RATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_sequences():
    # Create output dictionary
    results_dict = dict()

    for dataset in DATASETS:
        print(f'Dataset: {dataset.capitalize()}')
        results_dict[dataset] = dict()

        # Load input data and labels
        inputs, _ = load_data(dataset_name=dataset, fold='test')

        # Unpack the shape
        num_seq, seq_length, num_features = inputs.shape
        num_samples = num_seq*seq_length

        # Merge sequences into continous stream
        inputs = inputs.reshape(num_samples, num_features)
        
        deviations = []
        collection_rates = []
        errors = []
        for collection_rate in COLLECTION_RATES:
            # Make the policy
            policy = BudgetWrappedPolicy(name='uniform',
                                        num_seq=num_seq,
                                        seq_length=seq_length,
                                        num_features=num_features,
                                        dataset=dataset,
                                        collection_rate=collection_rate,
                                        collect_mode='tiny',
                                        window_size=20,
                                        model='',
                                        max_skip=0)
            max_num_seq = num_samples
            policy.init_for_experiment(num_sequences=max_num_seq)

            # Run the policy
            policy_result = run_policy(policy=policy,
                        sequence=inputs,
                        should_enforce_budget=True)


            collected_list = policy_result.measurements
            estimate_list = policy_result.estimate_list
            collected_indices = flatten(policy_result.collected_indices)
                        
            # Stack collected features into a numpy array
            collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]

            # Reconstruct the sequence
            reconstructed = reconstruct_sequence(measurements=collected,
                                                collected_indices=collected_indices,
                                                seq_length=num_samples)

            # Calculate errors
            mae = round(mean_absolute_error(y_true=inputs, y_pred=reconstructed), 3)

            errors.append(mae)
            deviations.append(policy.deviation)
            collection_rates.append(collection_rate)

        results_dict[dataset]['errors'] = errors
        results_dict[dataset]['deviations'] = deviations
        results_dict[dataset]['collection_rates'] = collection_rates


    with plt.style.context(PLOT_STYLE):
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=False)
        axs = axs.ravel()
        plt.subplots_adjust(hspace=0.5)

        for i, dataset in enumerate(DATASETS):
            axs[i].set_xticks(results_dict[dataset]['collection_rates'], results_dict[dataset]['errors'])
        
            axs[i].plot(results_dict[dataset]['collection_rates'], results_dict[dataset]['deviations'], label='Deviation', linestyle='dashed', dashes=(20, 5), color='blue')
            axs[i].scatter(results_dict[dataset]['collection_rates'], results_dict[dataset]['deviations'], marker='o', color='darkblue', alpha=0.5, zorder=2)

            axs[i].set_title(f'{dataset.capitalize()}', fontsize=16)
            axs[i].set_xlabel('Collection Rate (and Error)', fontsize=14)
            axs[i].set_ylabel('Deviation', fontsize=14)
        
        fig.tight_layout()
        plt.show()

    # Save results
    output_file = os.path.join('results/', 'dev.json.gz')
    save_json_gz(results_dict, output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    run_sequences()