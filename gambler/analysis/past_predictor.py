import json
import os
import pandas as pd
import numpy as np
import random
import pickle
import statistics
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statistics import geometric_mean

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse, mean_absolute_percentage_error
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.analysis import normalized_rmse
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten
from gambler.utils.constants import DATASETS, WINDOW_SIZES
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.analysis.plot_utils import bar_plot, COLORS, PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH, MARKER, MARKER_SIZE


# DATASETS = ['epilepsy']
COLLECTION_RATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_sequences():
    # Create output dictionary
    results_dict = dict()

    for dataset in DATASETS:
        print(f'Dataset: {dataset.capitalize()}')
        results_dict[dataset] = dict()

        # Load input data and labels
        inputs, _ = load_data(dataset_name=dataset, fold='test')

        # Unpack the shape
        num_seq, seq_length, num_features = inputs.shape
        num_samples = num_seq*seq_length

        # Merge sequences into continous stream
        inputs = inputs.reshape(num_samples, num_features)

        # Load the prediction model
        # model_name = f'saved_models/{dataset}/models/gambler_model_20'
        # model = pickle.load(open(model_name, 'rb'))

        for collection_rate in COLLECTION_RATES:
            prediction_errors = []
            # results_dict[dataset][collection_rate] = dict()

            # Make the policy
            policy_name = 'adaptive_train'
            policy = BudgetWrappedPolicy(name=policy_name,
                                        num_seq=num_seq,
                                        seq_length=seq_length,
                                        num_features=num_features,
                                        dataset=dataset,
                                        collection_rate=collection_rate,
                                        collect_mode='tiny',
                                        window_size=20,
                                        model='',
                                        max_skip=0)
            max_num_seq = num_samples
            policy.init_for_experiment(num_sequences=max_num_seq)

            # Run the policy
            _ = run_policy(policy=policy,
                        sequence=inputs,
                        should_enforce_budget=True)

            y_pred_dev = []
            y_pred_rate = []
            for dev, _, ratio in policy.training_data:
                # pred = model.predict(np.array([dev, collection_rate]).reshape(1, -1))[0]
                # y_pred_rate.append(pred)
                y_pred_rate.append(ratio)
                y_pred_dev.append(dev)

            y_pred_dev = y_pred_dev[:-1]
            y_pred_rate = y_pred_rate[:-1]
            
            y_true_dev = [dev for dev, _, _ in policy.training_data][1:]
            y_true_rate = [ratio for _, _, ratio in policy.training_data][1:]

            # Calculate prediction error
            dev_r2 = r2_score(y_true=y_true_dev, y_pred=y_pred_dev)
            rate_r2 = r2_score(y_true=y_true_rate, y_pred=y_pred_rate)
            prediction_errors.append([dev_r2, rate_r2])
            results_dict[dataset][collection_rate] = prediction_errors

            # results_dict[dataset][collection_rate]['y_pred'] = y_pred
            # results_dict[dataset][collection_rate]['y_true'] = y_true

    # Save results
    output_file = os.path.join('results/past_predictor', 'past_predictor.json.gz')
    save_json_gz(results_dict, output_file)

        
def run_timeseries():
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-type', type=str, choices=['timeseries', 'sequences'])
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)
    
    if args.dataset_type == 'timeseries':
        run_timeseries()
    elif args.dataset_type == 'sequences':
        run_sequences()