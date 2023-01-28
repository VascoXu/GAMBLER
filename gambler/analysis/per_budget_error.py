import json
import os
import numpy as np
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from statistics import geometric_mean

from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

policies = {'uniform': 'Uniform',
            'adaptive_heuristic': 'Adaptive Heuristic',
            'adaptive_deviation': 'Adaptive Deviation',
            'adaptive_uniform': 'Adaptive Uniform',
            'adaptive_gambler': 'Adaptive Gambler',
            }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    # Load the results
    dist_num = 1
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'{args.distribution}_distribution_{dist_num}.json.gz')
    filepath = os.path.join(base, '../results', f'budget_savings.json.gz')
    dataset_results = read_json_gz(filepath)

    dataset = 'epilepsy'

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE[0]*0.75, PLOT_SIZE[1]))    

        for policy in policies.keys():
            per_budget_errors = dataset_results[dataset][policy]
            errors = [error[1] for error in per_budget_errors]
            collection_rates = [rate[0] for rate in per_budget_errors]
            ax.plot(collection_rates, errors, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=policies[policy], color=COLORS[policy])

        ax.set_xlabel('Collection Rates', fontsize=AXIS_FONT)
        ax.set_ylabel('MAE', fontsize=AXIS_FONT)
        ax.set_title(f'{dataset.capitalize()} Dataset', fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        fig.tight_layout()
        plt.show()

