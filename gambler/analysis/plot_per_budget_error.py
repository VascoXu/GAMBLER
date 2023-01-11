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

from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data


DEVIATION_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
HEURISTIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_heuristic --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
GAMBLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_gambler --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=[])
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--num-runs', type=int, default=20)
    parser.add_argument('--distribution', type=str, default='\'\'')
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=100)
    args = parser.parse_args()

    policies = {'uniform': UNIFORM_CMD, 
                'heuristic': HEURISTIC_CMD, 
                'deviation': DEVIATION_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_gambler': GAMBLER_CMD,
                }

    NUM_RUNS = args.num_runs

    # Seed for reproducible results
    random.seed(3485)

    # Load data
    fold = 'train'
    inputs, labels = load_data(dataset_name=args.dataset, fold=fold, dist='')
    num_seq = len(labels)
    labels = list(set(labels.reshape(-1)))
    num_labels = len(labels)

    # Run policies on different budgets and distributions
    errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_gambler': []}
    policy_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_gambler': []}
    final_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_gambler': []}
    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for budget in budgets:
        for policy in policies.keys():
            errors[policy].append([])

            for i in range(NUM_RUNS):
                # Generate fixed-length string of random classes
                seq_left = num_seq
                classes = []
                while seq_left > 0:
                    rand_label = labels[random.randrange(len(labels))]
                    rand_length = random.randint(0, seq_left)
                    seq_left -= rand_length
                    classes += [rand_label for _ in range(rand_length)]            

                cmd = policies[policy].format(args.dataset, budget, args.window_size, args.distribution)
                classes_lst = classes
                if len(classes) > 0:
                    classes = ' '.join([str(label) for label in classes_lst])
                    cmd += f' --classes {classes}'

                res = run_command(cmd)
                try:
                    error, num_collected, total_samples = res.split(',')
                    errors[policy][-1].append(float(error))
                except:
                    print(res)

    print(errors)