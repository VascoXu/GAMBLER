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
            'adaptive_gambler': 'Adaptive Gambler',
            }


def get_pct_dec(dist_num):
   # Load the results
    distribution = 'skewed'
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'{distribution}_distribution_{dist_num}.json.gz')
    dataset_results = read_json_gz(filepath)

    # Compute mean error of each dataset
    mean_errors = dict()
    mean_diffs = dict()
    for dataset in dataset_results.keys():
        mean_errors[dataset] = []
        mean_diffs[dataset] = []

    for dataset in dataset_results.keys():
        uniform_errors = [error[1] for error in dataset_results[dataset]['uniform']]

        for policy in policies.keys():
            errors = [error[1] for error in dataset_results[dataset][policy]]

            if args.normalized:
                 # normalize error to uniform
                errors = np.divide(errors, uniform_errors)

            # compute average
            mean_error = round(sum(errors)/len(errors), 4)
            mean_uniform = round(sum(uniform_errors)/len(uniform_errors), 4)
            pct_diff = round(((mean_uniform-mean_error)/mean_error)*100, 4)
        
            mean_errors[dataset].append(mean_error)
            mean_diffs[dataset].append(pct_diff)

    # Compute median percent error higher than Uniform sampling
    diffs = []
    for i in range(len(policies.keys())):
        tmp = []
        for dataset in dataset_results.keys():
            tmp.append(mean_diffs[dataset][i])
        diffs.append(round(statistics.median(tmp), 4))
    diffs = [str(diff) for diff in diffs]

    return diffs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    args = parser.parse_args()
    
    NUM_K = 5

    k_diffs = dict()
    for policy in policies.keys():
        k_diffs[policy] = []

    for i in range(1, NUM_K+1):
        diffs = get_pct_dec(i)
        for j, policy in enumerate(policies.keys()):
            k_diffs[policy].append(float(diffs[j]))

    for policy in policies.keys():
        k_diffs[policy].append(statistics.mean(k_diffs[policy]))

    # Plot mean errors
    ks = [f'k={i}' for i in range(1, NUM_K+1)] + ['Overall']
    legend = [policies[policy] for policy in policies.keys()]
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))

        xs = bar_plot(ax, k_diffs, total_width=.8, single_width=1, colors=list(COLORS.values()), legend=legend)

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)
        
        plt.ylabel('Error Reduction Compared to Uniform Sampling (%)', fontsize=18)
        plt.xlabel('Randomization Parameter', fontsize=18)
        plt.xticks(range(len(ks)), ks, fontsize=AXIS_FONT)
        plt.title(f'Reconstruction Error across Multiple Random Distributions', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()