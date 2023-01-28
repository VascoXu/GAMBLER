import json
import os
import numpy as np
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from statistics import geometric_mean

from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    window_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'window_sizes_experiment_{args.distribution}.json.gz')
    results = read_json_gz(filepath)

    with plt.style.context(PLOT_STYLE):
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=False)
        axs = axs.ravel()
        plt.subplots_adjust(hspace=0.5)

        for i, dataset in enumerate(list(results.keys())[1:]):
            
            avg_errors = []
            for window_size in results[dataset]:
                errors = [error[1] for error in results[dataset][window_size]]
                avg_errors.append(np.mean(errors))


            axs[i].plot(range(len(avg_errors)), avg_errors)
            axs[i].set_xticks(range(len(avg_errors)), window_sizes, fontsize=AXIS_FONT)
            axs[i].set_title(f'{dataset.capitalize()}', fontsize=TITLE_FONT)
            axs[i].set_xlabel('Window Size')
            axs[i].set_ylabel('Reconstruction Error (MAE)')
        
        fig.suptitle(f'{args.distribution.capitalize()}: Window Size vs Reconstruction Error', fontsize=16)
        fig.tight_layout()
        # plt.savefig(f'graphs/window_size_{args.distribution}.png', bbox_inches='tight')
        plt.show()