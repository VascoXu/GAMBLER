import json
import os
import numpy as np
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from statistics import geometric_mean

from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

policies = {'uniform': 'Uniform',
            'adaptive_heuristic': 'Adaptive Heuristic',
            'adaptive_deviation': 'Adaptive Deviation',
            'adaptive_budget': 'Adaptive Budget',
            'adaptive_gambler': 'Adaptive Gambler',
            }

policies = {'adaptive_gambler': 'Adaptive Gambler'}
# policies = {'adaptive_deviation': 'Adaptive Deviation'}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    # Load the results
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'savings_{args.distribution}.json.gz')
    dataset_results = read_json_gz(filepath)

    # Compute mean error of each dataset
    mean_savings = dict()
    for policy in policies.keys():
        mean_savings[policy] = []

    for dataset in dataset_results.keys():
        uniform_savings = [error[1] for error in dataset_results[dataset]['uniform']]
        uniform_mean = geometric_mean(uniform_savings)

        for policy in policies.keys():
            savings = [error[1] for error in dataset_results[dataset][policy]]
            print(policy, savings)

            # compute policy average
            # policy_mean = sum(savings)/len(savings)
            policy_mean = geometric_mean(savings)
            
            # compute percentage decrease
            mean_diff = uniform_mean-policy_mean
            pct_dec = (mean_diff/uniform_mean)*100

            # normalize savings to uniform
            # normalized_mean = policy_mean/uniform_mean
            mean_savings[policy].append(pct_dec)


    # Compute mean of datasets
    for policy in policies.keys():
        policy_mean = statistics.mean(mean_savings[policy])
        mean_savings[policy].append(policy_mean)

    # Plot mean errors
    datasets = [dataset for dataset in dataset_results.keys()] + ['Overall']
    legend = [policies[policy] for policy in policies.keys()]
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = [color for dataset,color in COLORS.items() if dataset in policies.keys()]

        xs = bar_plot(ax, mean_savings, total_width=.8, single_width=1, colors=colors, legend=legend)

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)

        plt.ylabel('Energy Reduction compared to Uniform (%)', fontsize=16)
        plt.xlabel('Datasets', fontsize=16)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        plt.title(f'Energy Savings across Multiple Datasets (Distribution: {args.distribution.capitalize()})', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()
