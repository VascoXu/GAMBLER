import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools

from gambler.analysis.plot_utils import bar_plot

ADAPTIVE_STATIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_CONTROLLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_controller --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_GREEDY_CMD = 'python run_policy.py --dataset {0} --policy adaptive_greedy --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BANDIT_CMD = 'python run_policy.py --dataset {0} --policy adaptive_bandit --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BUDGET_CMD = 'python run_policy.py --dataset {0} --policy adaptive_budget --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_BUDGET_SIGMA_CMD = 'python run_policy.py --dataset {0} --policy adaptive_budget_sigma --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'


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
    res.append('Random')
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

    distributions = ['\'\'', '0', '1', '2', '3', 'rand']
    dist_labels = get_distribution_str(dist, num_labels)

    policies = {'uniform': UNIFORM_CMD, 
                'deviation': ADAPTIVE_STATIC_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_budget': ADAPTIVE_BUDGET_CMD,
                'adaptive_budget_sigma': ADAPTIVE_BUDGET_SIGMA_CMD,
                }

    dists = get_distribution_str(dist, num_labels)

    # Run policies on different distributions
    errors = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_budget_sigma': []}
    for dist in distributions:
        for policy in policies.keys():
            if dist == 'rand':
                cmd = policies[policy].format(args.dataset, args.collection_rate, args.window_size, 'even')
                cmd += ' --randomize group --block-size 1'
            else:
                cmd = policies[policy].format(args.dataset, args.collection_rate, args.window_size, dist)

            res = run_command(cmd)
            print(res)
            error, num_collected, total_samples = res.split(',')
            errors[policy].append(float(error))

    # # Calculate average error
    for policy in policies.keys():
        errors[policy].append(sum(errors[policy])/len(errors[policy]))

    fig, ax = plt.subplots()
    print(errors)
    bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Threshold', 'Adaptive Uniform', 'Adaptive Bandit', 'Adaptive Sigma'])
    plt.xticks(range(len(dist_labels)), dist_labels)
    plt.title(f'Average Label Error {args.collection_rate}')
    plt.show()