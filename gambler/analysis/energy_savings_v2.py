#!/bin/python3

from importlib.metadata import distribution
import os.path
import numpy as np
import time
from collections import defaultdict, namedtuple
from argparse import ArgumentParser
from typing import List
from collections import Counter
import random
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.constants import SMALL_NUMBER, BIG_NUMBER
from gambler.utils.loading import load_data
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten


BatchResult = namedtuple('BatchResult', ['mae', 'num_collected'])
MAX_ITER = 30  # Prevents any unexpected infinite looping


def execute_policy(policy: BudgetWrappedPolicy, batch: np.ndarray, energy_margin: float) -> BatchResult:    
    # Unpack the shape
    seq_length, num_features = batch.shape

    policy.init_for_experiment(num_sequences=seq_length)
    policy.reset()

    policy_result = run_policy(policy=policy, sequence=batch, should_enforce_budget=False)

    # Reconstruct the sequence elements, [T, D]
    if len(policy_result.measurements) > 0:
        collected_list = policy_result.measurements
        collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]
        reconstructed = reconstruct_sequence(measurements=collected,
                                            collected_indices=flatten(policy_result.collected_indices),
                                            seq_length=seq_length)            
    else:
        reconstructed = np.zeros([policy.window_size, num_features])

    # Compute the error over the batch
    error = mean_absolute_error(y_true=inputs, y_pred=reconstructed)

    return BatchResult(mae=error, num_collected=policy_result.num_collected)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--randomize', type=str, default='expected')
    parser.add_argument('--rand-amount', type=int, default=1)
    parser.add_argument('--num-runs', type=int, default=20)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    policies = [
        'uniform',
        'adaptive_heuristic',
        'adaptive_deviation',
        'adaptive_uniform',
        'adaptive_budget',
        'adaptive_gambler',
    ]

    policies = [
        'uniform',
        'adaptive_deviation',
    ]

    window_sizes = {
        'epilepsy': 10,
        'uci_har': 50, 
        'wisdm': 20,
        'trajectories': 50,
        'pedestrian': 20, 
        'temperature': 20,
        'pavement': 30, 
        'haptics': 5,
        'eog': 50
    }    

    # Seed for reproducible results
    random.seed(42)

    collection_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    NUM_RUNS = len(collection_rates) if args.randomize == 'expected' else args.num_runs

    # Build dictionary to hold results
    savings_map = dict()
    for dataset in args.datasets:
        savings_map[dataset] = dict()
        for policy in policies:
            savings_map[dataset][policy] = []

    for dataset in args.datasets:
        # Load the data
        true_inputs, true_labels = load_data(dataset, fold='test')
        unique_labels = list(set(true_labels))

        # Get the maximum threshold value based on the data
        max_threshold = np.max(np.sum(np.abs(true_inputs), axis=-1)) + 1000.0        

        # Unpack the data dimensions
        num_seq, seq_length, num_features = true_inputs.shape
        length = num_seq*seq_length

        for i in range(NUM_RUNS):
            if args.randomize == 'skewed':
                # Select random budget
                budget = random.choice(collection_rates)
            else:
                # Select collection rate
                budget = collection_rates[i]

            # Generate random distribution
            classes = []
            if args.randomize == 'skewed':
                seq_left = num_seq
                while seq_left > 0:
                    rand_label = unique_labels[random.randrange(len(unique_labels))]
                    rand_length = max(1, random.randint(0, seq_left//args.rand_amount))
                    seq_left -= rand_length
                    classes += [rand_label for _ in range(rand_length)]       

            # print(*classes, sep=' ') 
    
            # Rearrange data for experiments
            inputs, labels = get_data(classes, true_inputs, true_labels)

            # Unpack the shape
            num_seqs, seq_lengths, num_features = inputs.shape

            # Merge sequences into continous stream
            inputs = inputs.reshape(num_seqs*seq_lengths, num_features)

            # Unpack the shape
            seq_length, num_features = inputs.shape

            # Create the the baseline policy (uniform)
            policy = BudgetWrappedPolicy(name='uniform',
                                        num_seq=num_seq,
                                        seq_length=seq_length,
                                        num_features=num_features,
                                        dataset=dataset,
                                        collection_rate=budget,
                                        collect_mode='tiny',
                                        window_size=20,
                                        model='',
                                        max_skip=0)                                    

            # Execute uniform policy
            policy_results = execute_policy(policy=policy, batch=inputs, energy_margin=0)
            uniform_mae = policy_results.mae
            uniform_samples = policy_results.num_collected
            savings_map[dataset]['uniform'].append([budget, uniform_samples])

            if args.should_print:
                print(f'DATASET: {dataset} | COLLECTION RATE: {budget}')
                print('==========')
                print(f'POLICY UNIFORM: {uniform_mae:.5f} | BUDGET: {uniform_samples}')
                print('==========')

            for policy_name in policies[1:]:
                if args.should_print:
                    print('POLICY:', policy_name)
                
                # Set initial search parameters
                best_mae = 10000
                best_budget = 10000
                iter_count = 0

                # Set the lower threshold based on the model type
                lower = -1 * max_threshold
                upper = max_threshold

                # Create the policy for which to fit thresholds
                policy = BudgetWrappedPolicy(name=policy_name,
                                            num_seq=num_seq,
                                            seq_length=seq_length,
                                            num_features=num_features,
                                            dataset=dataset,
                                            collection_rate=budget,
                                            collect_mode='tiny',
                                            window_size=window_sizes[dataset],
                                            model='',
                                            max_skip=0)

                # Perform a "binary-search" for most comparable-budget that achieve similar
                # accuracy to uniform.
                while (iter_count < MAX_ITER) and (lower < upper):
                    # Set the current threshold
                    current = (upper + lower) / 2

                    policy.init_for_experiment(num_sequences=inputs.shape[0]*inputs.shape[1])
                    policy.reset()

                    # Set the threshold for the policy
                    policy.set_threshold(threshold=current)

                    # Execute policy
                    policy_results = execute_policy(policy=policy, batch=inputs, energy_margin=0)
                    mae = policy_results.mae
                    num_collected = policy_results.num_collected

                    if args.should_print:
                        print(f'MAE: {mae:0.5f} | # Collected: {num_collected}, Threshold: {current}', end='\r')
                        # print(f'Best MAE: {mae:0.5f} | Best Collected: {num_collected}, Best Budget: {current}, Lower: {lower}, Upper: {upper}')

                    if mae > uniform_mae:
                        upper = current # Decrease given budget
                    else:
                        lower = current # Increase given budget

                    # Ensure that lower <= upper
                    temp = min(lower, upper)
                    upper = max(lower, upper)
                    lower = temp
                    iter_count += 1

                
                savings_map[dataset][policy_name].append([budget, num_collected])
                print('\n==========')


    output_file = os.path.join('results', 'savings.json.gz')
    save_json_gz(savings_map, output_file)