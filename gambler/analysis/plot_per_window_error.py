from tkinter import font
import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from gambler.utils.cmd_utils import run_command
from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

DEVIATION_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window-size {2} --should-enforce-budget'
HEURISTIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_heuristic --collection-rate {1} --window-size {2} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window-size {2} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window-size {2} --should-enforce-budget'
GAMBLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_gambler --collection-rate {1} --window-size {2} --should-enforce-budget'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policies', type=str, nargs='+', required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    policies = {'uniform': UNIFORM_CMD, 
                'heuristic': HEURISTIC_CMD, 
                'deviation': DEVIATION_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'adaptive_gambler': GAMBLER_CMD,
                }

    chosen_policies = ['uniform', 'adaptive_heuristic']

    # Execute selected policies
    for policy in chosen_policies:
        cmd = policies[args.policy].format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
        res = run_command(cmd)

    collection_rates = []
    for policy in args.policies:
        res = read_json_gz(f'results/{policy}-tiny_{round(args.collection_rate*100)}.json.gz')
        rates = res['collection_ratios']
        collection_rates.append(rates)

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * 0.75))

        xs = list(range(len(collection_rates[0])))
        ax1.plot(xs, rates[0], label='Uniform Policy', linewidth=LINE_WIDTH)
        ax1.plot(xs, rates[1], label='Adaptive Policy', linewidth=LINE_WIDTH)

        ax1.set_xlabel('Window Number', fontsize=AXIS_FONT)
        ax1.set_ylabel('Collection Rate', fontsize=AXIS_FONT)

        ax1.set_title('Collection Rate per Window', fontsize=TITLE_FONT)

        """
        plt.axvline(x=13, color='black', linestyle='dashed')
        plt.axvline(x=28, color='black', linestyle='dashed')
        plt.axvline(x=43, color='black', linestyle='dashed')
        plt.axvline(x=55, color='black', linestyle='dashed')

        plt.axvline(x=14, color='black', linestyle='dashed')
        plt.axvline(x=29, color='black', linestyle='dashed')
        plt.axvline(x=97, color='black', linestyle='dashed')
        plt.axvline(x=112, color='black', linestyle='dashed')

        ax1.text(0.56, 0.05, s='Adapt #: {0}'.format(deviation['num_collected']), fontsize=LEGEND_FONT, transform=ax1.transAxes)
        ax1.text(0.78, 0.05, s='Adapt Error: {0:.3f}'.format(deviation['mae']), fontsize=LEGEND_FONT, transform=ax1.transAxes)

        ax1.text(0.56, 0.1, s='Uniform #: {0}'.format(policy['num_collected']), fontsize=LEGEND_FONT, transform=ax1.transAxes)
        ax1.text(0.78, 0.1, s='Uniform Error: {0:.3f}'.format(policy['mae']), fontsize=LEGEND_FONT, transform=ax1.transAxes)
        """

        ax1.legend()
        plt.show()

