
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import h5py
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz

TRAIN_CMD = 'python3 run_policy.py --dataset {0} --policy adaptive_train --collection-rate {1} --window-size {2} --distribution {3} --fold {4} --train --should-enforce-budget'


if __name__ == '__main__':
    """Join training and testing datasets"""
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=100)
    args = parser.parse_args()

    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Remove file, if exists
    train_filename = f'train/{args.dataset}/train.csv'
    val_filename = f'train/{args.dataset}/test.csv'

    """
    try:
        os.remove(train_filename)
        os.remove(val_filename)
    except OSError:
        pass

    # Generate training set
    for budget in budgets:
        cmd = TRAIN_CMD.format(args.dataset, budget, args.window_size, '\'\'', 'train')
        print(cmd)
        res = run_command(cmd)

    print("Done Generating Training Dataset")

    # Generate validation set
    for budget in budgets:
        cmd = TRAIN_CMD.format(args.dataset, budget, args.window_size, '\'\'', 'test')
        print(cmd)
        res = run_command(cmd)

    print("Done Generating Validation Dataset")

    print("Train Random Forest")
    """

    # Load training data
    training_set = pd.read_csv(train_filename)
    X_train = training_set.iloc[:, 0:2].values.tolist()
    y_train = training_set.iloc[:, 2:].values.ravel()

    X_train = X_train[:-1]
    y_train = y_train[1:]

    # Load validation set
    validation_set = pd.read_csv(val_filename)
    X_val = training_set.iloc[:, 0:2].values.tolist()
    y_val = training_set.iloc[:, 2:].values.ravel()

    X_val = X_val[:-1]
    y_val = y_val[1:]

    print("Finished Partition")
    
    model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
    # model = RandomForestRegressor(n_estimators=5, max_depth=7, random_state=42).fit(X_train, y_train)
    # model = RandomForestRegressor(n_estimators=3, max_depth=8, random_state=42).fit(X_train, y_train)

    """
    Best parameters for individual datasets
    
    Pavement
    {'max_depth': 8, 'n_estimators': 3}
    """

    # # Grid search for best parameters
    # param_grid = { 
    #     'n_estimators': [3, 5, 10, 30, 50, 100],
    #     'max_depth' : [4,5,6,7,8],
    # }

    # CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
    # CV_rfc.fit(X_train, y_train)
    # print(CV_rfc.best_params_)

    print(model.score(X_val, y_val))
    pickle.dump(model, open(f'saved_models/{args.dataset}/random_forest/rf', 'wb'))

