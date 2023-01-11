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
            'adaptive_uniform': 'Adaptive Uniform',
            'adaptive_budget': 'Adaptive Budget',
            'adaptive_prob': 'Adaptive Prob',
            'adaptive_gambler': 'Adaptive Gambler',
            }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    # Load the results
    dist_num = 3
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'{args.distribution}_distribution_{dist_num}.json.gz')
    windowpath = os.path.join(base, '../results', f'window_sizes_experiment_{args.distribution}.json.gz')
    dataset_results = read_json_gz(filepath)

    # Compute mean error of each dataset
    mean_errors = dict()
    mean_std = dict()
    for policy in policies.keys():
        mean_errors[policy] = []
        mean_std[policy] = []

    for dataset in dataset_results.keys():
        uniform_error = [error[1] for error in dataset_results[dataset]['uniform']]

        for policy in dataset_results[dataset].keys():
            errors = [error[1] for error in dataset_results[dataset][policy]]

            if args.normalized:
                 # normalize error to uniform
                errors = np.divide(errors, uniform_error)
            
            # compute average
            # mean_error = sum(errors)/len(errors)
            mean_error = geometric_mean(errors)
            mean_errors[policy].append(mean_error)
            mean_std[policy].append(np.std(errors))


    # Compute mean of datasets
    for policy in policies.keys():
        # policy_mean = sum(mean_errors[policy])/len(mean_errors[policy])
        policy_mean = geometric_mean(mean_errors[policy])
        mean_errors[policy].append(policy_mean)
        mean_std[policy].append(np.std(mean_errors[policy]))


    # Plot mean errors
    datasets = [dataset for dataset in dataset_results.keys()] + ['Overall']
    legend = [policies[policy] for policy in policies.keys()]
    dist = 'Random' if args.distribution == 'skewed' else 'Expected'
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))

        xs = bar_plot(ax, mean_errors, error=mean_std, total_width=.8, single_width=1, colors=list(COLORS.values()), legend=legend)

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)
        
        plt.ylabel('MAE (Normalized to Uniform)', fontsize=16)
        plt.xlabel('Datasets', fontsize=16)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        plt.title(f'Reconstruction Error across Multiple Datasets (Distribution: {dist} [k={dist_num}])', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()
