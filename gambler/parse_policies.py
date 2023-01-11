import json
import numpy as np
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter

from gambler.analysis.plot_utils import bar_plot
from gambler.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

policies = ['uniform',
            'heuristic',
            'deviation',
            'adaptive_uniform',
            'adaptive_budget',
            'adaptive_gambler',
            ]

extensions = {
    'skewed': '100',
    'skewed-budget': '100',
    '.': '100',
    'expected': 'exp',
    'expected-budget': 'exp'
}

budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 'Mean Error']
datasets = ['epilepsy', 'uci_har', 'pavement', 'trajectories', 'haptics', 'eog', 'temperature', 'wisdm', 'pedestrian', 'mean']
datasets = ['epilepsy', 'uci_har', 'pavement', 'trajectories', 'eog', 'temperature', 'wisdm', 'pedestrian', 'mean']
legend = ['Adaptive Heuristic', 'Adaptive Deviation', 'Adaptive Uniform', 'Gambler']
legend = ['Adaptive Heuristic', 'Adaptive Deviation', 'Adaptive Uniform', 'Adaptive Budget', 'Gambler']


def rewrite_file(filename):
    # Replace single-quotes with double-quotes for JSON parsing
    line = ''
    with open(f'{filename}.txt', "rt") as fin:
        line = fin.readline()
        out = line.replace("'", '"').rstrip()

    with open(f'{filename}.txt', "wt") as fout:
        fout.write(out)


def plot_normalized_error(foldername):
    ext = extensions[foldername]
    mean_errors = {'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_gambler': []}
    mean_errors = {'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_gambler': []}

    for dataset in datasets[:-1]:
        rewrite_file(f'{foldername}/{dataset}_{ext}')
        with open(f'{foldername}/{dataset}_{ext}.txt') as f:
            results = json.load(f)

            # Normalize to uniform error
            a = np.array(results['uniform'])
            if 'expected' not in foldername:
                uniform_error = [error[1] for error in a]
            else:
                uniform_error = a[0]

            # Calculate mean error across random budgets
            for policy in policies:
                if policy != 'uniform':
                    a = np.array(results[policy])
                    if 'expected' not in foldername:
                        a = [error[1] for error in a]
                    else:
                        a = a[0]
                    normalized_error = [0 if uniform_error[i] == 0 else a[i]/uniform_error[i] for i in range(len(a))]
                    mean_errors[policy].append(statistics.geometric_mean(normalized_error))


    for policy in policies:
        if policy != 'uniform':
            mean_errors[policy].append(statistics.geometric_mean(mean_errors[policy]))

    with plt.style.context('seaborn-ticks'):             
        fig, ax = plt.subplots()
        bar_plot(ax, mean_errors, total_width=.8, single_width=.9, legend=legend)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        # plt.title(f'Normalized Error across Multiple Datasets ({foldername.capitalize()})', fontsize=TITLE_FONT)
        plt.title(f'Normalized Error across Multiple Datasets (Skewed)', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()        


def plot_reconstruction_error(foldername):
    ext = extensions[foldername]
    box_data = [[], [], [], [], [], []]
    mean_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_gambler': []}
    mean_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_gambler': []}

    for dataset in datasets:
        rewrite_file(f'{foldername}/{dataset}_{ext}')
        with open(f'{foldername}/{dataset}_{ext}.txt') as f:
            results = json.load(f)

            # Calculate mean error across random budgets
            for idx,policy in enumerate(policies):
                a = np.array(results[policy])
                if 'expected' not in foldername:
                    a = [error[1] for error in a]
                else:
                    a = a[0]
                mean_errors[policy].append([np.mean(a).tolist()])
                box_data[idx] = a

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()
        legend.insert(0, 'Uniform')

        # Creating plot
        # plt.boxplot(box_data)
        # plt.xticks([1, 2, 3, 4, 5], legend)

        bar_plot(ax, mean_errors, total_width=.8, single_width=.9, legend=legend)
        plt.xticks(range(len(datasets)), datasets, fontsize=AXIS_FONT)
        # plt.title(f'Reconstruction Error across Multiple Datasets ({foldername.capitalize()})', fontsize=TITLE_FONT)
        plt.title(f'Reconstruction Error across Multiple Datasets (Skewed)', fontsize=TITLE_FONT)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--error', type=str, default='reconstruction')
    parser.add_argument('--foldername', type=str, default='.')
    args = parser.parse_args()

    if args.error == 'reconstruction':
        plot_reconstruction_error(args.foldername)
    elif args.error == 'normalized':
        plot_normalized_error(args.foldername)
