from tkinter import font
import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from gambler.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from gambler.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

ADAPTIVE_STATIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --output-folder {4}'
ADAPTIVE_CONTROLLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_controller --collection-rate {1} --window {2} --distribution {3} --output-folder {4}'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --output-folder {4}'
ADAPTIVE_GREEDY_CMD = 'python run_policy.py --dataset {0} --policy adaptive_greedy --collection-rate {1} --window {2} --distribution {3} --output-folder {4} --should-enforce-budget'
ADAPTIVE_BANDIT_CMD = 'python run_policy.py --dataset {0} --policy adaptive_bandit --collection-rate {1} --window {2} --distribution {3} --output-folder {4} --should-enforce-budget'

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
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--randomize', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--distribution', type=str, default='')
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--window-size', type=int, default=0)
    args = parser.parse_args()

    dist = -1
    policy = ''

    adaptive_static_cmd = ADAPTIVE_STATIC_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
    res = run_command(adaptive_static_cmd)

    if args.policy == 'adaptive_controller':
        adaptive_cmd = ADAPTIVE_CONTROLLER_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
        res = run_command(adaptive_cmd)
        policy = 'adaptive_controller'
    elif args.policy == 'adaptive_greedy':
        adaptive_cmd = ADAPTIVE_GREEDY_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
        res = run_command(adaptive_cmd)
        policy = 'adaptive_greedy'
    elif args.policy == 'adaptive_bandit':
        adaptive_cmd = ADAPTIVE_BANDIT_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
        res = run_command(adaptive_cmd)
        policy = 'adaptive_bandit'
    elif args.policy == 'uniform':
        uniform_cmd = UNIFORM_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution, 'results/')
        res = run_command(uniform_cmd)
        policy = 'uniform'
    
    deviation = read_json_gz(f'results/adaptive_deviation-tiny_{round(args.collection_rate*100)}.json.gz')
    deviation_rates = deviation['collection_ratios']
    policy = read_json_gz(f'results/{policy}-tiny_{round(args.collection_rate*100)}.json.gz')
    policy_rates = policy['collection_ratios']

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * 0.75))

        xs = list(range(len(deviation_rates)))
        ax1.plot(xs, policy_rates, label='Uniform Policy', linewidth=LINE_WIDTH)
        ax1.plot(xs, deviation_rates, label='Adaptive Policy', linewidth=LINE_WIDTH)

        ax1.set_xlabel('Window Number', fontsize=AXIS_FONT)
        ax1.set_ylabel('Collection Rate', fontsize=AXIS_FONT)

        ax1.set_title('Collection Rate per Window', fontsize=TITLE_FONT)

        # plt.axvline(x=13, color='black', linestyle='dashed')
        # plt.axvline(x=28, color='black', linestyle='dashed')
        # plt.axvline(x=43, color='black', linestyle='dashed')
        # plt.axvline(x=55, color='black', linestyle='dashed')

        # plt.axvline(x=14, color='black', linestyle='dashed')
        # plt.axvline(x=29, color='black', linestyle='dashed')
        # plt.axvline(x=97, color='black', linestyle='dashed')
        # plt.axvline(x=112, color='black', linestyle='dashed')

        # ax1.text(0.56, 0.05, s='Adapt #: {0}'.format(deviation['num_collected']), fontsize=LEGEND_FONT, transform=ax1.transAxes)
        # ax1.text(0.78, 0.05, s='Adapt Error: {0:.3f}'.format(deviation['mae']), fontsize=LEGEND_FONT, transform=ax1.transAxes)

        # ax1.text(0.56, 0.1, s='Uniform #: {0}'.format(policy['num_collected']), fontsize=LEGEND_FONT, transform=ax1.transAxes)
        # ax1.text(0.78, 0.1, s='Uniform Error: {0:.3f}'.format(policy['mae']), fontsize=LEGEND_FONT, transform=ax1.transAxes)

        ax1.legend()
        plt.show()

