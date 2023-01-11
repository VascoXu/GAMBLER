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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num-runs', type=int, default=100)
    args = parser.parse_args()

    scores = {'uniform': [], 'adaptive_deviation': [], 'adaptive_heuristic': [], 'adaptive_uniform': [], 'adaptive_budget': [], 'adaptive_gambler': []}
    policies = ['uniform', 'adaptive_heuristic', 'adaptive_deviation', 'adaptive_uniform', 'adaptive_budget', 'adaptive_gambler']
    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    NUM_RUNS = args.num_runs
    WINDOW_SIZE = 20
    MODELNAME = f'saved_models/{args.dataset}/{args.dataset}.sav'

    # Seed for reproducible results
    random.seed(42)

    # Load data
    fold = 'train'
    inputs, train_labels = load_data(dataset_name=args.dataset, fold=fold, dist='')
    num_seq, seq_length, num_features = inputs.shape
    train_length = len(train_labels)
    labels = list(set(train_labels.reshape(-1)))
    num_labels = len(labels)

    # Load model for dataset
    model = pickle.load(open(MODELNAME, 'rb'))

    for _ in range(NUM_RUNS):
        # Select random budget
        rand_budget = random.choice(budgets)

        # Generate random distribution
        seq_left = train_length
        classes = []
        while seq_left > 0:
            rand_label = train_labels[random.randrange(len(labels))]
            rand_length = random.randint(0, seq_left)
            seq_left -= rand_length
            classes += [rand_label for _ in range(rand_length)]

        for policy_name in policies:
            policy = BudgetWrappedPolicy(name=policy_name,
                                        num_seq=num_seq,
                                        seq_length=seq_length*num_seq,
                                        num_features=num_features,
                                        dataset=args.dataset,
                                        collection_rate=rand_budget,
                                        collect_mode='tiny',
                                        window_size=WINDOW_SIZE,
                                        model='',
                                        max_skip=0)

            # Load the data
            inputs, labels = load_data(dataset_name=args.dataset, fold='test', dist='')
            labels = labels.reshape(-1)

            # Unpack the shape
            num_seq, seq_length, num_features = inputs.shape

            # Rearrange data for experiments
            args.classes = classes
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

            # Reshape merged sequence
            reconstructed = reconstructed.reshape((num_seq, seq_length*num_features))

            # Run reconstructed sequence through Classifier
            acc = model.score(reconstructed, labels)

            scores[policy_name].append([acc, mae, norm_mae, rand_budget])


    print(scores)