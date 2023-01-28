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
import statistics
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.constants import SMALL_NUMBER, BIG_NUMBER, WINDOW_SIZES
from gambler.utils.loading import load_data
from gambler.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz
from gambler.utils.data_manager import get_data
from gambler.utils.misc_utils import flatten

BatchResult = namedtuple('BatchResult', ['error', 'num_collected', 'did_exhaust'])

EPSILON = 1e-4
TOLERANCE = 1e-2
MAX_ITER = 100

BATCHES_PER_TRIAL = 1
BATCH_SIZE = 256

def generate_random_distribution(labels, num_seq, rand_amount):
    classes = []

    seq_left = num_seq
    while seq_left > 0:
        rand_label = labels[random.randrange(num_labels)]
        rand_length = max(1, random.randint(0, seq_left//rand_amount))
        seq_left -= rand_length
        classes += [rand_label for _ in range(rand_length)]

    return classes


def execute_policy(policy: BudgetWrappedPolicy, batch: np.ndarray, should_enforce_budget: bool=False) -> BatchResult:
    policy.init_for_experiment(num_sequences=batch.shape[0])

    # Execute the policy on each sequence
    estimated_list: List[np.ndarray] = []

    num_collected = 0
    for idx, sequence in enumerate(batch):
        policy.reset()

        policy_result = run_policy(policy=policy, sequence=sequence, should_enforce_budget=should_enforce_budget, reconstruct=False)

        # Reconstruct the sequence elements, [T, D]
        if len(policy_result.measurements) > 0:
            collected_list = policy_result.measurements
            collected_indices = flatten(policy_result.collected_indices)
            collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([seq_length, num_features])  # [K, D]
            reconstructed = reconstruct_sequence(measurements=collected,
                                                collected_indices=collected_indices,
                                                seq_length=seq_length)   

            num_collected += len(collected_indices)

        else:
            reconstructed = np.zeros([seq_length, num_features])

        estimated_list.append(np.expand_dims(reconstructed, axis=0))

    # Compute the error over the batch
    estimated = np.vstack(estimated_list)  # [B, T, D]
    error = np.average(np.abs(batch - estimated))

    return BatchResult(error=error, num_collected=num_collected, did_exhaust=policy.has_exhausted_budget())


def fit_threshold(policy: BudgetWrappedPolicy,
        inputs: np.ndarray,
        collection_rate: int,
        should_print: bool,):

    num_seq, seq_length, num_features = inputs.shape
    num_samples = num_seq*seq_length
    sample_idx = np.arange(num_seq)

    rand = np.random.RandomState(seed=581)

    # Get the maximum threshold value based on the data
    max_threshold = np.max(np.sum(np.abs(inputs), axis=-1)) + 1000.0        

    # Set the lower threshold based on the model type
    lower = -1 * max_threshold
    upper = max_threshold

    best_threshold = upper
    best_error = BIG_NUMBER

    batch_size = min(len(sample_idx), BATCH_SIZE)

    # No need to run multiple batches when we are already
    # using the full data-set
    batches_per_trial = 1 if batch_size == len(sample_idx) else BATCHES_PER_TRIAL

    # Set the budget for the policy
    # policy.set_budget(seq_length*batch_size*batches_per_trial*collection_rate)
    policy.set_budget(seq_length*num_seq*collection_rate)

    iter_count = 0
    while (iter_count < MAX_ITER) and (abs(upper - lower) > EPSILON):
        
        current = (upper + lower) / 2

        # Set the threshold for the policy
        policy.set_threshold(threshold=current)

        # Make lists to track results
        did_exhaust_list: List[bool] = []
        error_list: List[float] = []

        num_collected = 0
        for _ in range(batches_per_trial):
            # Make the batch
            batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
            batch = inputs[batch_idx]

            batch_result = execute_policy(policy=policy, batch=inputs)

            error_list.append(batch_result.error)
            did_exhaust_list.append(batch_result.did_exhaust)
            num_collected += batch_result.num_collected

        # Get aggregate metrics
        error = np.average(error_list)
        did_exhaust = any(did_exhaust_list)

        # Track the best error
        if (error < best_error) and (not did_exhaust):
            best_threshold = current
            best_error = error

        if should_print:
            print('Error: {0:.7f}, Threshold: {1:.7f}'.format(error, current), end='\r')

        # Get the search direction based on the budget use
        if did_exhaust:
            lower = current  # Reduce the energy consumption
        else:
            upper = current  # Increase the energy consumption

        # Ensure that lower <= upper
        temp = min(lower, upper)
        upper = max(lower, upper)
        lower = temp
        iter_count += 1

    if args.should_print:
        print('')

    return (best_threshold, best_error)


def fit_adaptive_budget(policy: BudgetWrappedPolicy, inputs: np.ndarray, dataset: str, collection_rate: float, target_error: float):
    """
    Perform a "binary-search" for most comparable-budget that achieve similar accuracy to GAMBLER.
    """

    # Load training data
    train_inputs, _ = load_data(dataset_name=dataset, fold='validation')
    val_inputs, _ = load_data(dataset_name=dataset, fold='train')

    num_seq, seq_length, num_features = train_inputs.shape
    num_samples = inputs.shape[0]*inputs.shape[1]

    lower = 0
    upper = round((collection_rate*100) + ((collection_rate*100)*0.5)) # use heuristic to speed up iteration
    current = 0.0
    observed_error = BIG_NUMBER

    best_observed = 0.0
    best_diff = BIG_NUMBER
    best_budget = upper

    error_lst = []

    rate = collection_rate
    old_rate = 0

    # Create parameters for policy validation data splitting
    rand = np.random.RandomState(seed=3485)

    # Create the policy for which to fit thresholds
    test_policy = BudgetWrappedPolicy(name=policy_name,
                                num_seq=num_seq,
                                seq_length=seq_length,
                                num_features=num_features,
                                dataset=dataset,
                                collection_rate=collection_rate,
                                collect_mode='tiny',
                                window_size=20,
                                model='',
                                max_skip=0)
    test_policy.init_for_experiment(num_sequences=num_seq)
    test_policy.reset()    

    # while (abs(observed_error - target_error) > EPSILON) or (best_observed > target_error):
    while (abs(rate - old_rate) > TOLERANCE) and (upper - lower) >= 1:
        current = (upper + lower) / 2
        
        old_rate = rate
        rate = round(current/100, 2)

        # Create the policy for which to fit thresholds
        policy = BudgetWrappedPolicy(name=policy_name,
                                    num_seq=num_seq,
                                    seq_length=seq_length,
                                    num_features=num_features,
                                    dataset=dataset,
                                    collection_rate=rate,
                                    collect_mode='tiny',
                                    window_size=20,
                                    model='',
                                    max_skip=0)
        policy.init_for_experiment(num_sequences=num_seq)
        policy.reset()        

        # Set the threshold for the policy
        threshold, _ = fit_threshold(policy=policy,
                        inputs=train_inputs,
                        collection_rate=rate,
                        should_print=args.should_print)

        # Execute policy on 'current' threshold
        test_policy.init_for_experiment(num_sequences=inputs.shape[0])
        test_policy.reset()
        test_policy.set_threshold(threshold=threshold)

        policy_results = execute_policy(policy=test_policy, batch=inputs)
        observed_error, num_collected = policy_results.error, policy_results.num_collected

        if args.should_print:
            print(f'Error: {observed_error:0.5f} | # Collected: {num_collected} | Budget: {collection_rate*num_samples} | Threshold: {threshold} | Rate: {rate}', end='\r')

        if observed_error < target_error:
            upper = current # Decrease given budget
        else:
            lower = current # Increase given budget

        diff = abs(target_error - observed_error)
        if (diff < best_diff) and (observed_error <= target_error):
            best_observed = observed_error
            best_diff = diff

        error_lst.append([observed_error, num_collected])

        temp = min(lower, upper)
        upper = max(lower, upper)
        lower = temp

        if args.should_print:
            print('')
    
    _, idx = min((abs(val[0] - target_error), idx) for (idx, val) in enumerate(error_lst))
    closest_error, closest_collected = error_lst[idx][0], error_lst[idx][1]

    return (closest_error, closest_collected)


def fit_uniform_budget(policy: BudgetWrappedPolicy, inputs: np.ndarray, target_error: float):
    """
    Perform a "binary-search" for most comparable-budget that achieve similar accuracy to GAMBLER.
    """

    num_seq, seq_length, num_features = inputs.shape
    num_samples = num_seq*seq_length

    lower = 0
    upper = 100
    current = 0.0
    observed_error = BIG_NUMBER

    best_observed = 0.0
    best_diff = BIG_NUMBER
    best_budget = upper

    error_lst = []

    while (abs(observed_error - target_error) > EPSILON) or (best_observed > target_error):

        if (upper - lower) < 1e-1:
            break

        current = (upper + lower) / 2
        rate = current/100
        
        policy = BudgetWrappedPolicy(name='uniform',
                                    num_seq=num_seq,
                                    seq_length=seq_length,
                                    num_features=num_features,
                                    dataset=dataset,
                                    collection_rate=rate,
                                    collect_mode='tiny',
                                    window_size=20,
                                    model='',
                                    max_skip=0)
        policy.init_for_experiment(num_sequences=num_seq)
        policy.reset()

        # Execute policy
        policy_results = execute_policy(policy=policy, batch=inputs)
        observed_error, num_collected = policy_results.error, policy_results.num_collected

        if args.should_print:
            # print(f'Error: {observed_error:0.5f} | # Collected: {num_collected} | Budget: {rate*num_samples} | Rate: {rate}', end='\r')
            print(f'Error: {observed_error:0.5f} | # Collected: {num_collected} | Budget: {rate*num_samples} | Rate: {rate} | Lower: {lower} Upper: {upper}')

        if observed_error < target_error:
            upper = current # Decrease given budget
        else:
            lower = current # Increase given budget

        diff = abs(target_error - observed_error)
        if (diff < best_diff) and (observed_error <= target_error):
            best_observed = observed_error
            best_diff = diff

        error_lst.append([observed_error, num_collected])

        # Ensure that lower <= upper
        temp = min(lower, upper)
        upper = max(lower, upper)
        lower = temp

    if args.should_print:
        print('')

    _, idx = min((abs(val[0] - target_error), idx) for (idx, val) in enumerate(error_lst))
    closest_error, closest_collected = error_lst[idx][0], error_lst[idx][1]

    return (closest_error, closest_collected)


def get_policy_stats(policy_results, test_inputs):
    # Unpack input shape
    num_seq, seq_length, num_features = test_inputs.shape
    num_samples = num_seq*seq_length
    test_inputs = test_inputs.reshape(num_seq*seq_length, num_features)

    # Stack collected features into a numpy array
    collected = np.vstack(policy_results.measurements) if len(policy_results.measurements) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]

    # Reconstruct the sequence
    collected_indices = flatten(policy_results.collected_indices)
    reconstructed = reconstruct_sequence(measurements=collected,
                                        collected_indices=collected_indices,
                                        seq_length=num_samples)

    # Calculate errors
    mae = mean_absolute_error(y_true=test_inputs, y_pred=reconstructed)  

    return (mae, policy_results.num_collected)  


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--distribution', type=str, default='expected')
    parser.add_argument('--rand-amount', type=int, default=1)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    # Initialize base parameters
    policies = [
        # 'adaptive_deviation',
        # 'adaptive_heuristic',
        'uniform',
    ]
    budgets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_runs = len(budgets)

    # Setupp output file
    savings_map = dict()

    # Build dictionary to hold results
    for dataset in args.datasets:
        if dataset not in savings_map:
            savings_map[dataset] = dict()
        for policy in policies:
            savings_map[dataset][policy] = []
        savings_map[dataset]['adaptive_gambler'] = []

    
    for dataset in args.datasets:
        # Load dataset
        inputs, labels = load_data(dataset_name=dataset, fold='train')
        unique_labels = list(set(labels))
        num_seq = len(inputs)
        num_labels = len(unique_labels)

        _test_inputs, _test_labels = load_data(dataset_name=dataset, fold='test')

        print(f"Dataset: {dataset}")
        
        # Run experiment for each policy
        for i in range(num_runs):
            print(f"Run # {i}")

            # Select collection rate
            budget = budgets[i] if args.distribution == 'expected' else random.choice(budgets)

            # Generate skewed/random distribution (if selected)
            if args.distribution == 'skewed':
                classes = generate_random_distribution(unique_labels, num_seq, args.rand_amount)
                test_inputs, test_labels = get_data(classes, _test_inputs, _test_labels)
            else:
                test_inputs, test_labels = _test_inputs, _test_labels

            # Unpack data shape
            num_seq, seq_length, num_features = test_inputs.shape
            num_samples = num_seq*seq_length

            # Execute GAMBLER policy (baseline)
            policy = BudgetWrappedPolicy(name='adaptive_gambler',
                                        num_seq=num_seq,
                                        seq_length=num_samples,
                                        num_features=num_features,
                                        dataset=dataset,
                                        collection_rate=budget,
                                        collect_mode='tiny',
                                        window_size=WINDOW_SIZES[dataset],
                                        model='',
                                        max_skip=0)
            policy_results = run_policy(policy=policy,
                                        sequence=test_inputs.reshape(num_seq*seq_length, num_features),
                                        reconstruct=False,
                                        should_enforce_budget=True)
            target_error, gambler_collected = get_policy_stats(policy_results, test_inputs)
            savings_map[dataset]['adaptive_gambler'].append([budget, gambler_collected, target_error])

            if args.should_print:
                print(f"Given Budget: {budget} | # Sample Budget: {budget*num_samples} | Target Error: {target_error} | Collected: {gambler_collected}")

            for policy_name in policies:
                print(f"Policy: {policy_name}")

                # Make the policy
                policy = BudgetWrappedPolicy(name=policy_name,
                                            num_seq=num_seq,
                                            seq_length=seq_length,
                                            num_features=num_features,
                                            dataset=dataset,
                                            collection_rate=budget,
                                            collect_mode='tiny',
                                            window_size=20,
                                            model='',
                                            max_skip=0)                

                if policy_name == 'uniform':
                    error, num_collected = fit_uniform_budget(policy, test_inputs, target_error)
                elif policy_name.startswith('adaptive'):
                    error, num_collected = fit_adaptive_budget(policy, test_inputs, dataset, budget, target_error)

                if args.should_print:
                    print(f"\nEnd Run: # Collected {num_collected} | MAE: {error}\n")

                # Store results
                savings_map[dataset][policy_name].append([budget, num_collected, error])

        # Save results
        foldername = 'results/budget_savings'
        filename = f'budget_savings_{args.distribution}_{dataset}.json.gz'
        output_file = os.path.join(foldername, filename)
        save_json_gz(savings_map, output_file)