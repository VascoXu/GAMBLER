import numpy as np
import pandas as pd
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statistics import geometric_mean

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.analysis import normalized_rmse
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH


def merge_sequence(sequence):
    # Merge feature dimensions
    merged = np.zeros((sequence.shape[0], 1))
    for i, sample in enumerate(sequence):
        x, y, z = sample
        merged[i] = ((x**2 + y**2 + z**2)**0.5)
    return merged


def read_data(filename):
    inputs = pd.read_csv(filename).to_numpy()
    return inputs


if __name__ == '__main__':
    # Seed for reproducible results
    random.seed(42)

    # Run policy
    filename = 'phone-accel/sitting_accel.csv'
    inputs = read_data(filename=filename)

    # Unpack the shape
    num_seqs, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = merge_sequence(inputs)

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots()
        ax1.set_ylim((0, 5))

        xs = list(range(len(inputs)))
        ax1.plot(xs, inputs[:, 0], label='True', color='red', linewidth=3, alpha=0.65, zorder=1)

        ax1.set_title('Sampling on Accelerometer Data', fontsize=16)
        ax1.set_xlabel('Time Step', fontsize=14)
        ax1.set_ylabel('Total Acceleration', fontsize=14)

        ax1.legend(loc='upper left')

        plt.show()