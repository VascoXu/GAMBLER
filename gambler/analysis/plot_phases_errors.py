import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command


ADAPTIVE_STATIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_CONTROLLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_controller --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_GREEDY_CMD = 'python run_policy.py --dataset {0} --policy adaptive_greedy --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BANDIT_CMD = 'python run_policy.py --dataset {0} --policy adaptive_bandit --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BUDGET_CMD = 'python run_policy.py --dataset {0} --policy adaptive_budget --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BUDGET_SIGMA_CMD = 'python run_policy.py --dataset {0} --policy adaptive_budget_sigma --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_PHASES_CMD = 'python run_policy.py --dataset {0} --policy adaptive_phases --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'


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
    res.append('Random')
    res.append('Mean')

    return res


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--randomize', type=str, default='')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    dist = 60
    num_labels = 4

    budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    policies = {'uniform': UNIFORM_CMD, 
                'deviation': ADAPTIVE_STATIC_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_budget': ADAPTIVE_BUDGET_CMD,
                'adaptive_budget_sigma': ADAPTIVE_BUDGET_SIGMA_CMD,
                'adaptive_phases': ADAPTIVE_PHASES_CMD
                }

    # Run policies on different distributions
    errors = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_budget_sigma': [], 'adaptive_phases': []}
    for budget in budgets:
        for policy in policies.keys():
            if dist == 'rand':
                cmd = policies[policy].format(args.dataset, budget, args.window_size, 'even')
                cmd += ' --randomize group --block-size 1'
            else:
                cmd = policies[policy].format(args.dataset, budget, args.window_size, 'even')

            res = run_command(cmd)
            print(res)
            error, num_collected, total_samples = res.split(',')
            errors[policy].append(float(error))

    # Calculate average error
    for policy in policies.keys():
        errors[policy].append(sum(errors[policy])/len(errors[policy]))

    fig, ax = plt.subplots()
    print(errors)
    bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Threshold', 'Adaptive Uniform', 'Adaptive Bandit', 'Adaptive Sigma', 'Adaptive Phases'])
    plt.xticks(range(len(budgets)), budgets)
    plt.title(f'Average Label Error')
    plt.show()