import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import h5py
import random
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import List, Tuple
from pickle import NONE
import pexpect

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from data_manager import get_data

ADAPTIVE_BANDIT_CMD = 'python3 run_policy.py --dataset {0} --policy adaptive_bandit --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
UNIFORM_CMD = 'python3 run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'


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
    args = parser.parse_args()

    args.dataset = 'epilepsy'
    args.window_size = 200
    args.collection_rate = 0.4
    args.distribution = 'even'

    # Load true data
    fold = 'test'
    inputs, labels = load_data(dataset_name=args.dataset, fold=fold, dist=args.distribution)
    num_seq, seq_length, num_features = inputs.shape
    labels = labels.reshape(-1)

    # Run uniform command
    cmd = UNIFORM_CMD.format(args.dataset, args.collection_rate, args.window_size, args.distribution)
    res = run_command(cmd)

    # Load uniform data
    fold = 'reconstructed'
    uniform_inputs, uniform_labels = load_data(dataset_name=args.dataset, fold=fold, dist=args.distribution, filename='reconstructed_uniform_even.h5')
    uniform_labels = uniform_labels.reshape(-1)

    # Error of individual label
    print('UNIFORM')
    total_error = 0
    for i in range(4):
        label_idx = np.where(labels == i)[0]
        idv_u_inputs = np.asarray([uniform_inputs[i] for i in label_idx])
        idv_t_inputs = np.asarray([inputs[i] for i in label_idx])

        idv_u_inputs = idv_u_inputs.reshape(-1, num_features)
        idv_t_inputs = idv_t_inputs.reshape(-1, num_features)

        label_error = mean_squared_error(y_true=idv_t_inputs, y_pred=idv_u_inputs)
        total_error += label_error
        print(f'Label {i} Error: {label_error}')
    print(f'Average Total Error: {total_error/4}')

    # Run adaptive bandit command
    cmd = ADAPTIVE_BANDIT_CMD.format(args.dataset, args.collection_rate, args.window_size, 'even')
    res = run_command(cmd)

    # Load bandit data
    fold = 'reconstructed'
    bandit_inputs, bandit_labels = load_data(dataset_name='epilepsy', fold=fold, dist=args.distribution, filename='reconstructed_adaptive_bandit_even.h5')
    bandit_labels = bandit_labels.reshape(-1)

    print()

    # Error of individual label
    print('BANDIT')
    total_error = 0
    for i in range(4):
        label_idx = np.where(labels == i)[0]
        idv_a_inputs = np.asarray([bandit_inputs[i] for i in label_idx])
        idv_t_inputs = np.asarray([inputs[i] for i in label_idx])

        idv_a_inputs = idv_a_inputs.reshape(-1, num_features)
        idv_t_inputs = idv_t_inputs.reshape(-1, num_features)

        label_error = mean_squared_error(y_true=idv_t_inputs, y_pred=idv_a_inputs)
        total_error += label_error
        print(f'Label {i} Error: {label_error}')
    print(f'Average Total Error: {total_error/4}')

    bandit_inputs = bandit_inputs.reshape(-1, num_features)
    uniform_inputs = uniform_inputs.reshape(-1, num_features)
    inputs = inputs.reshape(-1, num_features)

    uniform_error = mean_squared_error(y_true=inputs, y_pred=uniform_inputs)
    bandit_error = mean_squared_error(y_true=inputs, y_pred=bandit_inputs)

    print("Total Uniform Error: ", uniform_error)
    print("Total Bandit Error: ", bandit_error)