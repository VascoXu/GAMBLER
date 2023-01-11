import numpy as np
import pandas as pd
from pickle import NONE
import pexpect
from argparse import ArgumentParser
import csv 
import matplotlib.pyplot as plt
import itertools
import random
import pickle
import statistics
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score

from gambler.utils.data_utils import reconstruct_sequence
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data


WINDOW_SIZE = 20

def execute_policy(policy_name, budget, dataset):
    # Load data
    fold = 'test'
    inputs, labels = load_data(dataset_name=dataset, fold=fold, dist='')
    num_seq, seq_length, num_features = inputs.shape
    train_length = len(labels)
    labels = list(set(labels.reshape(-1)))
    num_labels = len(labels)

    policy = BudgetWrappedPolicy(name=policy_name,
                                num_seq=num_seq,
                                seq_length=seq_length*num_seq,
                                num_features=num_features,
                                dataset=dataset,
                                collection_rate=budget,
                                collect_mode='tiny',
                                window_size=WINDOW_SIZE,
                                model='',
                                max_skip=0)

    # Load the data
    inputs, labels = load_data(dataset_name=dataset, fold='test', dist='')
    labels = labels.reshape(-1)

    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    # Rearrange data for experiments
    inputs, labels = get_data(args, inputs, labels)

    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    # Merge sequences into continous stream
    inputs = inputs.reshape(inputs.shape[0]*seq_length, num_features)

    # Execute the policy
    policy_result = run_policy(policy=policy, sequence=inputs, should_enforce_budget=True)

    # Reconstruct sequence
    collected_list = policy_result.measurements
    collected_indices = policy_result.collected_indices

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]
    reconstructed = reconstruct_sequence(measurements=collected,
                                        collected_indices=collected_indices,
                                        seq_length=inputs.shape[0])

    # Calculate error of reconstruction
    mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)
    norm_mae = normalized_mae(y_true=inputs, y_pred=reconstructed)
    rmse = mean_squared_error(y_true=inputs, y_pred=reconstructed, squared=False)
    norm_rmse = normalized_rmse(y_true=inputs, y_pred=reconstructed)    

    print("mae: ", mae)

    return (reconstructed, labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, default='uniform')
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    # Load training data
    X_train, y_train = load_data(dataset_name=args.dataset, fold='train', dist='')
    num_seq, seq_length, num_features = X_train.shape
    X_train = X_train.reshape((num_seq, -1))

    # Load testing data
    X_test, y_test = load_data(dataset_name=args.dataset, fold='test', dist='')
    num_seq, seq_length, num_features = X_test.shape
    X_test = X_test.reshape((num_seq, -1))

    clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', n_estimators=500, random_state=42)
    # {'criterion': 'gini', 'max_depth': 7, 'max_features': 'log2', 'n_estimators': 200}
    # {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 500}
    # {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 200}
    # {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'log2', 'n_estimators': 500}
    # reconstructed, labels = execute_policy(args.policy, 0.7, args.dataset)
    # reconstructed = reconstructed.reshape(X_train.shape)

    # Grid search for best parameters
    """
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }

    clf = RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)
    """

    # # Train model
    clf.fit(X_train, y_train)

    # # Save model
    pickle.dump(clf, open(f'saved_models/{args.dataset}/{args.dataset}.sav' ,'wb'))

    y_pred = clf.predict(X_test)
    print("Test Accuracy: ", accuracy_score(y_test, y_pred))

