import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz

TRAIN_CMD = 'python3 run_policy.py --dataset {0} --policy adaptive_train --collection-rate {1} --window {2} --distribution {3} --fold {4} --should-enforce-budget'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=200)
    args = parser.parse_args()

    budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Remove training file if exists
    try:
        os.remove('data.csv')
    except OSError:
        pass

    # Training data
    for budget in budgets:
        cmd = TRAIN_CMD.format(args.dataset, budget, args.window_size, '\'\'', 'all')
        run_command(cmd)