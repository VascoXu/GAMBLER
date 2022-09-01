
from enum import unique
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
from sklearn.model_selection import train_test_split

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz

TRAIN_CMD = 'python run_policy.py --dataset {0} --policy adaptive_train --collection-rate {1} --window {2} --distribution {3} --fold {4} --train --classes {5} --should-enforce-budget'
ADAPTIVE_DEVIATION_CMD = 'python run_policy.py --dataset {0} --policy adaptive_deviation --collection-rate {1} --window {2} --distribution {3} --threshold {4} --should-enforce-budget'
ADAPTIVE_UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy adaptive_uniform --collection-rate {1} --window {2} --distribution {3} --threshold {4} --should-enforce-budget'
UNIFORM_CMD = 'python run_policy.py --dataset {0} --policy uniform --collection-rate {1} --window {2} --distribution {3} --should-enforce-budget'
GAMBLER_CMD = 'python run_policy.py --dataset {0} --policy adaptive_training --collection-rate {1} --window {2} --distribution {3} --model {4} --should-enforce-budget'


if __name__ == '__main__':
    """Join training and testing datasets"""
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    # Load testing dataset
    inputs, labels = load_data(dataset_name=args.dataset, fold='test', dist='')
    labels = labels.reshape(-1)

    classes = [str(label) for label in set(labels)]
    num_classes = len(classes)

    classes_dists = []
    for c in classes:
        myclass = classes[:]
        myclass.remove(c)
        classes_dists.append(' '.join([str(c) for c in myclass]))

    if args.train:
        """Train RandomForestRegression"""
        for c in classes:
            myclass = classes[:]
            myclass.remove(c)

            budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            # Remove file, if exists
            train_filename = f'train/{args.dataset}/train.csv'
            val_filename = f'train/{args.dataset}/validation.csv'
            try:
                os.remove(train_filename)
                os.remove(val_filename)
            except OSError:
                pass

            # Generate training set
            for budget in budgets:
                cmd = TRAIN_CMD.format(args.dataset, budget, args.window_size, '\'\'', 'train', ' '.join(myclass))
                res = run_command(cmd)

            # Generate validation set
            for budget in budgets:
                cmd = TRAIN_CMD.format(args.dataset, budget, args.window_size, '\'\'', 'validation', ' '.join(myclass))
                res = run_command(cmd)

            # Load training data
            training_set = pd.read_csv(train_filename)
            X_train = training_set.iloc[:, 0:2].values.tolist()
            y_train = training_set.iloc[:, 2:].values.ravel()

            # Load validation set
            validation_set = pd.read_csv(val_filename)
            X_val = training_set.iloc[:, 0:2].values.tolist()
            y_val = training_set.iloc[:, 2:].values.ravel()
            
            model = RandomForestRegressor(max_depth=7, random_state=0).fit(X_train, y_train)
            print(model.score(X_val, y_val))
            pickle.dump(model, open(f'saved_models/rf_{c}', 'wb'))

    
    policies = {'uniform': UNIFORM_CMD, 
                'adaptive_deviation': ADAPTIVE_DEVIATION_CMD, 
                'adaptive_uniform': ADAPTIVE_UNIFORM_CMD,
                'gambler': GAMBLER_CMD,
                }

    # # Run policies
    errors = {'uniform': [], 'adaptive_deviation': [], 'adaptive_uniform': [], 'gambler': []}
    for i, dists in enumerate(classes_dists):
        for policy in policies.keys():
            model_name = f'saved_models/{args.dataset}/random_forest/rf_{i}'
            base = os.path.dirname(__file__)
            threshold_name = os.path.join(base, 'saved_models', args.dataset, f'thresholds_block_{i}.json.gz')

            if policy == 'gambler':
                cmd = policies[policy].format(args.dataset, args.collection_rate, args.window_size, '\'\'', model_name)
            elif policy.startswith('adaptive'):
                cmd = policies[policy].format(args.dataset, args.collection_rate, args.window_size, '\'\'', threshold_name)
            else:
                cmd = policies[policy].format(args.dataset, args.collection_rate, args.window_size, '\'\'')
            
            res = run_command(cmd)
            print(cmd, res)
            error, num_collected, total_samples = res.split(',')
            errors[policy].append(float(error))

    # Calculate average error
    for policy in policies.keys():
        errors[policy].append(sum(errors[policy])/len(errors[policy]))

    fig, ax = plt.subplots()
    print(errors)
    bar_plot(ax, errors, total_width=.8, single_width=.9, legend=['Uniform', 'Adaptive Deviation', 'Adaptive Uniform', 'Gambler'])
    plt.xticks(range(len(classes_dists)), classes_dists)
    plt.title(f'Average Label Error {args.collection_rate}')
    plt.show()
