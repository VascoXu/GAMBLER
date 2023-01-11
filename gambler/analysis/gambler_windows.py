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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    window_sizes = {
        'epilepsy': 10,
        'uci_har': 50, 
        'wisdm': 20,
        'trajectories': 50,
        'pedestrian': 20, 
        'temperature': 20,
        'pavement': 30, 
        'haptics': 5,
        'eog': 50
    }    

    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'window_sizes_experiment_{args.distribution}.json.gz')
    results = read_json_gz(filepath)
    
    # Build results
    res = dict()
    res['static'] = []
    res['trained'] = []
    for i,dataset in enumerate(results.keys()):
        optimal_window = window_sizes[dataset]
        error_trained = statistics.mean([error[1] for error in results[dataset][str(optimal_window)]])
        error_static = statistics.mean([error[1] for error in results[dataset][str(20)]])
        res['static'].append(error_static/error_static)
        res['trained'].append(error_trained/error_static) 

    res['static'].append(statistics.mean(res['static']))
    res['trained'].append(statistics.mean(res['trained']))
        
    # Plot mean errors
    datasets = [dataset for dataset in results.keys()] + ['Overall']
    legend = ['Static Window', 'Trained Window']
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))

        xs = bar_plot(ax, res, total_width=.8, single_width=1, legend=legend)

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)

        plt.title(f'Trained Window Size vs Static Window Size (Distribution: {args.distribution.capitalize()})', fontsize=16)
        plt.xlabel('Datasets', fontsize=16)
        plt.ylabel(f'Normalized Error to Static Window Size', fontsize=TITLE_FONT)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        fig.tight_layout()
        plt.show()
