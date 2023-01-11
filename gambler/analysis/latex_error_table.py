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
            'adaptive_gambler': 'Adaptive Gambler',
            }


if __name__ == '__main__':
    """Print reconstruction error for each dataset for use in LATEX tables"""
    
    parser = ArgumentParser()
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--distribution', type=str, default='expected')
    args = parser.parse_args()

    # Load the results
    if args.distribution == 'expected':
        dist_num = 1
    else:
        dist_num = 1
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, '../results', f'{args.distribution}_distribution_{dist_num}.json.gz')
    windowpath = os.path.join(base, '../results', f'window_sizes_experiment_{args.distribution}.json.gz')
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

    # Print for LATEX table
    for dataset in dataset_results.keys():
        errors = [str(error) for error in mean_errors[dataset]] 
        print(' & '.join(errors))
    print(' & '.join(diffs))

    