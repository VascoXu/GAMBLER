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
ADAPTIVE_TRAINING_CMD = 'python run_policy.py --dataset {0} --policy adaptive_training --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'


def run_command(cmd):
    try:
        policy_obj = NONE
        policy_obj = pexpect.spawn(cmd)
        policy_obj.expect(pexpect.EOF)
        response = policy_obj.before.decode("utf-8").strip()
        return response
    finally:
        policy_obj.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--classes', type=int, nargs='+', required=True)
    parser.add_argument('--randomize', type=str, default='')
    parser.add_argument('--distribution', type=int, default='\'\'')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    dist = 60
    num_labels = 4

    distributions = ['']

    policies = {'uniform': UNIFORM_CMD, 
                'deviation': ADAPTIVE_STATIC_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_budget': ADAPTIVE_BUDGET_CMD,
                'adaptive_budget_sigma': ADAPTIVE_BUDGET_SIGMA_CMD,
                'adaptive_training': ADAPTIVE_TRAINING_CMD,
                }

    # Run policies on different distributions
    errors = {'uniform': [], 'deviation': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_budget_sigma': [], 'adaptive_training': []}
    budgets = [0.7]
    classes = ' '.join([str(label) for label in args.classes])
    for budget in budgets:
        for policy in policies.keys():
            cmd = policies[policy].format(args.dataset, budget, args.window_size, args.distribution)
            cmd += f' --labels test --classes {classes}'

            res = run_command(cmd)
            print(res)
            error, num_collected, total_samples = res.split(',')
            errors[policy].append(float(error))

    # # Calculate average error
    # for policy in policies.keys():
    #     errors[policy].append(sum(errors[policy])/len(errors[policy]))

    fig, ax = plt.subplots()
    print(errors)
    bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Threshold', 'Adaptive Uniform', 'Adaptive Bandit', 'Adaptive Sigma', 'Adaptive Training'])
    plt.xticks(range(len(budgets)), budgets)
    plt.title(f'Average Label Error {args.collection_rate}')
    plt.show()
