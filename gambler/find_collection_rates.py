import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz

ADAPTIVE_STATIC_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=200)
    args = parser.parse_args()

    budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    output_folder = f'saved_models/{args.dataset}/'

    result_dict = {}

    for budget in budgets:
        cmd = ADAPTIVE_STATIC_CMD.format(args.dataset, budget, args.window_size, 'even')
        res = run_command(cmd)
        collection_rates = res.split('\n')
        collection_rates = [float(cr.rstrip()) for cr in collection_rates]

        result_dict[budget] = collection_rates

    output_path = os.path.join(output_folder, 'collection_rates_validation.json.gz')
    save_json_gz(result_dict, output_path)
