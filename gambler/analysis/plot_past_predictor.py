import json
import os
import numpy as np
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from statistics import geometric_mean

from gambler.utils.constants import DATASETS, COLLECTION_RATES
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH


policies = {'uniform': 'Uniform',
            'adaptive_heuristic': 'Adaptive Heuristic',
            'adaptive_deviation': 'Adaptive Deviation',
            'adaptive_gambler': 'Adaptive Gambler',
            }

def load_results():
    # Load the results
    filename = 'past_predictor'
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, f'../results/{filename}', f'{filename}.json.gz')
    results = read_json_gz(filepath)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()
    
    # Load prediction results
    results = load_results()

    dataset_results = {'pred': []}
    for i, dataset in enumerate(DATASETS):
        stat = []
        for stats in results[dataset].values():
            stat.append(stats[0][1])
        dataset_results['pred'].append(statistics.mean(stat))

    # Plot mean errors
    datasets = DATASETS
    legend = ['Predicted Collection Rate']
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = [color for dataset,color in COLORS.items() if dataset in policies.keys()]

        xs = bar_plot(ax, dataset_results, total_width=.8, single_width=1, colors=colors, legend=legend)

        plt.ylabel('MAE for Predicted Collection Rate', fontsize=16)
        plt.xlabel('Datasets', fontsize=16)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        plt.title(f'Error for Predicted Collection Rate Across Datasets (Distribution: {args.distribution.capitalize()})', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()


    # with plt.style.context(PLOT_STYLE):
    #     fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=False)
    #     axs = axs.ravel()
    #     plt.subplots_adjust(hspace=0.5)

    #     for i, dataset in enumerate(DATASETS):
            
    #         stat = []
    #         for stats in results[dataset].values():
    #             stat.append(stats[0][0])

    #         axs[i].plot(range(len(stat)), stat)
    #         axs[i].set_xticks(range(len(stat)), COLLECTION_RATES, fontsize=AXIS_FONT)
    #         axs[i].set_title(f'{dataset.capitalize()}', fontsize=TITLE_FONT)
    #         axs[i].set_xlabel('Collection Rate')
    #         axs[i].set_ylabel('MAE for Predicted Dev')
        
    #     fig.suptitle(f'{args.distribution.capitalize()}: Predicted Deviation Error', fontsize=16)
    #     fig.tight_layout()
    #     # plt.savefig(f'graphs/window_size_{args.distribution}.png', bbox_inches='tight')
    #     plt.show()
