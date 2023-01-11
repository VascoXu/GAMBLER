import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools
import random
import statistics
from collections import Counter

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.loading import load_data
from gambler.utils.cmd_utils import run_command

DEVIATION_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window-size {2} --should-enforce-budget'
HEURISTIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_heuristic --collection-rate {1} --window-size {2} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window-size {2} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window-size {2} --should-enforce-budget'
ADAPTIVE_BUDGET_CMD = 'python run_policy.py --dataset {0} --policy adaptive_budget --collection-rate {1} --window-size {2} --should-enforce-budget'
GAMBLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_gambler --collection-rate {1} --window-size {2} --should-enforce-budget'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=[])
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--num-runs', type=int, default=20)
    parser.add_argument('--dist', type=str, default='normal')
    parser.add_argument('--distribution', type=str, default='\'\'')
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=100)
    args = parser.parse_args()

    policies = {'uniform': UNIFORM_CMD, 
                'heuristic': HEURISTIC_CMD, 
                'deviation': DEVIATION_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_budget': ADAPTIVE_BUDGET_CMD,
                'adaptive_gambler': GAMBLER_CMD,
                }

    NUM_RUNS = args.num_runs

    # Load data
    fold = 'train'
    inputs, labels = load_data(dataset_name=args.dataset, fold=fold, dist='')
    num_seq = len(labels)
    labels = list(set(labels.reshape(-1)))
    num_labels = len(labels)

    # Run policies on different budgets and distributions
    errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_gambler': []}
    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in range(NUM_RUNS):        
        for policy in policies.keys():
            errors[policy].append([])

        
        for budget in budgets:
            # Execute policy
            for policy in policies.keys():
                cmd = policies[policy].format(args.dataset, budget, args.window_size)
                res = run_command(cmd)
                try:
                    error, num_collected, total_samples = res.split(',')
                    errors[policy][i].append(float(error))
                except:
                    print(res)

    print(errors)
