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


from gambler.analysis.plot_utils import bar_plot

DEVIATION_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
HEURISTIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_heuristic --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'
ADAPTIVE_TRAINING_CMD = 'python run_policy.py --dataset {0} --policy adaptive_training --collection-rate {1} --window-size {2} --distribution {3} --should-enforce-budget'

def run_command(cmd):
    try:
        policy_obj = NONE
        policy_obj = pexpect.spawn(cmd, timeout=None)
        policy_obj.expect(pexpect.EOF)
        response = policy_obj.before.decode("utf-8").strip()
        return response
    finally:
        policy_obj.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--classes', type=int, nargs='+', default=[])
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--distribution', type=str, default='\'\'')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    policies = {'uniform': UNIFORM_CMD, 
                'heuristic': HEURISTIC_CMD, 
                'deviation': DEVIATION_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_training': ADAPTIVE_TRAINING_CMD,
                }

    NUM_RUNS = 100
    dist = 'binomial'

    # Run policies on different budgets and distributions
    errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_training': []}
    policy_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_training': []}
    final_errors = {'uniform': [], 'deviation': [], 'heuristic': [], 'adaptive_uniform': [], 'adaptive_training': []}
    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in range(NUM_RUNS):
        
        for policy in policies.keys():
            errors[policy].append([])

        for budget in budgets:
            # Generate random classes
            labels = [0, 1, 2, 3]

            if dist == 'normal':
                classes = [labels[random.randrange(len(labels))] for _ in range(25)]
            elif dist == 'binomial':
                dists = [0.2, 0.45, 0.55, 0.78]
                classes = np.random.binomial(n=3, p=random.choice(dists), size=25)

            # Execute policy
            for policy in policies.keys():
                cmd = policies[policy].format(args.dataset, budget, args.window_size, args.distribution)
                if len(args.classes) > 0 or args.randomize == True:
                    classes = ' '.join([str(label) for label in classes])
                    cmd += f' --classes {classes}'

                res = run_command(cmd)
                error, num_collected, total_samples = res.split(',')
                errors[policy][i].append(float(error))

    print(errors)