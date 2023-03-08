from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import h5py
import os
import csv
from typing import List, Tuple
import random
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


def train_gambler(dataset, window_size, fold):
    # List of collection rates
    collection_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Load the data
    inputs, labels = load_data(dataset_name=dataset, fold=fold)
    labels = labels.reshape(-1)

    # Unpack the shape
    num_seqs, seq_lengths, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

    # Unpack the shape
    seq_length, num_features = inputs.shape

    for collection_rate in collection_rates:
        # Make the policy
        collect_mode = 'tiny'
        policy_name = 'adaptive_train'
        policy = BudgetWrappedPolicy(name=policy_name,
                                    num_seq=num_seqs,
                                    seq_length=seq_length,
                                    num_features=num_features,
                                    dataset=dataset,
                                    collection_rate=collection_rate,
                                    collect_mode=collect_mode,
                                    window_size=window_size,
                                    model='',
                                    max_skip=0)

        max_num_seq = num_seqs

        policy.init_for_experiment(num_sequences=max_num_seq)

        # Run the policy
        _ = run_policy(policy=policy,
                       sequence=inputs,
                       should_enforce_budget=True)
    
        with open(f'train/{dataset}/{fold}_{window_size}.csv', 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(policy.training_data)


if __name__ == '__main__':
    """Train GAMBLER"""

    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--should-retrain', action='store_true')
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    for dataset in args.datasets:
        train_filename = f'train/{dataset}/train_{args.window_size}.csv'
        val_filename = f'train/{dataset}/val_{args.window_size}.csv'
        test_filename = f'train/{dataset}/test_{args.window_size}.csv'

        if args.should_retrain:
            try:
                os.remove(train_filename)
                os.remove(test_filename)
            except OSError:
                pass

            train_gambler(dataset, args.window_size, 'train')
            train_gambler(dataset, args.window_size, 'test')

        # Load training data
        training_set = pd.read_csv(train_filename)
        X_train = training_set.iloc[:, 0:2].values.tolist()
        y_train = training_set.iloc[:, 2:].values.ravel()

        # Load testing data
        testing_set = pd.read_csv(test_filename)
        X_test = testing_set.iloc[:, 0:2].values.tolist()
        y_test = testing_set.iloc[:, 2:].values.ravel()

        # Train Decision Tree Regressor
        model = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train, y_train)
        pickle.dump(model, open(f'saved_models/{dataset}/models/gambler_model_{args.window_size}', 'wb'))

        if args.should_print:
            print(f'{dataset.capitalize()} {args.window_size}')
            print('Model accuracy: ', model.score(X_test, y_test))