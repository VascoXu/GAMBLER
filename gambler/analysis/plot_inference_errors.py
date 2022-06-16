import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from gambler.analysis.plot_utils import bar_plot, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

ADAPTIVE_STATIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_CONTROLLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_controller --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_GREEDY_CMD = 'python run_policy.py --dataset {0} --policy adaptive_greedy --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
TRAIN_CLF_CMD = 'python train_datasets.py --dataset {0} --distribution {1}'

def run_command(cmd):
    try:
        policy_obj = NONE
        policy_obj = pexpect.spawn(cmd)
        policy_obj.expect(pexpect.EOF)
        response = policy_obj.before.decode("utf-8").strip()
        return response
    finally:
        policy_obj.close()


def get_distribution_str(dist, num_labels):
    p = round((100-dist) / (num_labels-1))
    dist_labels = []
    for i in range(num_labels-1):
        dist_labels.append(str(p))
    
    res = []
    for i in range(num_labels):
        temp = dist_labels.copy()
        temp.insert(i, str(dist))
        res.append('(' + "-".join(temp) + ')')

    res.insert(0, 'Even')
    res.append('Mean')

    return res


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--randomize', type=str, default='')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    dist = 60
    num_labels = 4

    distributions = ['even', 0, 1, 2, 3]
    dist_labels = get_distribution_str(dist, num_labels)

    policies = {'uniform': UNIFORM_CMD, 
                'deviation': ADAPTIVE_STATIC_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'controller': ADAPTIVE_CONTROLLER_CMD, 
                'greedy': ADAPTIVE_GREEDY_CMD
                }

    dists = get_distribution_str(dist, num_labels)

    # Run policies on different distributions
    # errors = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'controller': [], 'greedy': []}

    rates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # error_in_rates = {'uniform': [], 'deviation': [], 'greedy': []}
    error_in_rates = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'controller': [], 'greedy': []}
    errors = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'controller': [], 'greedy': []}
    for rate in rates:
        for dist in distributions:
            for policy in policies.keys():
                cmd = policies[policy].format(args.dataset, rate, args.window_size, dist)
                run_command(cmd)

                # Parse result
                res = run_command(TRAIN_CLF_CMD.format(args.dataset, dist))
                error = res.split(' ')[1]
                errors[policy].append(float(error))

        # Calculate average error
        for policy in policies.keys():
            errors[policy].append(sum(errors[policy])/len(errors[policy]))
            error_in_rates[policy].append(sum(errors[policy])/len(errors[policy]))

    # for error in error_in_rates.values():
    #     print(sum(error)/len(error))

    fig, ax = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * 0.75))
    ax.set_ylabel('Accuracy', fontsize=AXIS_FONT)
    ax.set_xlabel('Budgets', fontsize=AXIS_FONT)
    # bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Static', 'Adaptive Uniform', 'Adaptive Controller', 'Adaptive Greedy'])
    # bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'LiteSense', 'Gambler'])
    bar_plot(ax, error_in_rates, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Static', 'Adaptive Uniform', 'Adaptive Controller', 'Adaptive Greedy'])
    plt.xticks(range(len(rates)), rates)
    plt.title('Inference Accuracy', fontsize=TITLE_FONT)
    plt.show()