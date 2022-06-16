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

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from data_manager import get_data


def write_dataset(inputs, labels, filename, output_folder):
    # Write new dataset to h5 file
    # partition_folder = os.path.join('datasets', output_folder)
    partition_folder = os.path.join('./datasets', output_folder)
    if not os.path.exists(partition_folder):
        os.mkdir(partition_folder)

    partition_inputs = inputs
    partition_output = labels
    with h5py.File(os.path.join(partition_folder, f'{filename}.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
        input_dataset.write_direct(partition_inputs)

        output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
        output_dataset.write_direct(partition_output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--epsilon', type=float, default=0.0)           # used for debugging
    parser.add_argument('--randomize', type=str, default='')            # used for debugging 
    parser.add_argument('--block-size', type=int, default=1)            # used for debugging
    parser.add_argument('--labels', type=str, default='all')            # used for debugging
    parser.add_argument('--distribution', type=str, default='even')         # used for debugging
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--save-dataset', action='store_true')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--window-size', type=int, default=0)
    parser.add_argument('--max-skip', type=int, default=0)
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)

    # Load the data
    fold = 'test'
    inputs, labels = load_data(dataset_name=args.dataset, fold=fold, dist=args.distribution)
    labels = labels.reshape(-1)

    # Rearrange data for experiments
    inputs, labels = get_data(args, inputs, labels)

    # Unpack the shape (again to account for possible modifications in dataset)
    num_seq, seq_length, num_features = inputs.shape

    # Make the policy
    collection_rate = args.collection_rate

    policy = BudgetWrappedPolicy(name=args.policy,
                                 num_seq=num_seq,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 dataset=args.dataset,
                                 collection_rate=args.collection_rate,
                                 collect_mode='tiny',
                                 max_window_size=args.window_size,
                                 should_compress=False,
                                 epsilon=args.epsilon,
                                 max_skip=args.max_skip)

    errors: List[float] = []
    measurements: List[np.ndarray] = []
    estimate_list: List[np.ndarray] = []
    collected: List[List[int]] = []
    collected_counts = defaultdict(list)
    
    collection_ratios: List[List[float]] = []
    window_labels: List[int] = []

    max_num_seq = num_seq if args.max_num_samples is None else min(num_seq, args.max_num_samples)

    policy.init_for_experiment(num_sequences=max_num_seq)

    collected_seq = 1           # The number of sequences collected under the budget
    previous_label = -1         # Track the current label

    curr_window_size = 0        # The current window size of the controller
    collected_within_window = 0 # The current window size of the controller

    for idx, (sequence, label) in enumerate(zip(inputs, labels)):
        if idx >= max_num_seq:
            break

        policy.reset()

        # Reset parameters on label change
        if label != previous_label:
            policy.reset_params()    
        previous_label = label

        # Pack information about window size
        window = (collected_within_window, curr_window_size)

        # Run the policy
        policy_result = run_policy(policy=policy,
                                   sequence=sequence,
                                   window=window,
                                   seq_num=idx,
                                   should_enforce_budget=args.should_enforce_budget
                                   )
        policy.step(seq_idx=idx, count=policy_result.num_collected)

        # Reconstruct the sequence
        if policy_result.num_collected == 0:
            reconstructed = np.zeros(sequence.shape)
        else:
            reconstructed = reconstruct_sequence(measurements=policy_result.measurements,
                                                collected_indices=policy_result.collected_indices,
                                                seq_length=seq_length)

        error = mean_absolute_error(y_true=sequence, y_pred=reconstructed)

        if policy_result.num_collected > 0:
            collected_seq = idx + 1
        else:
            error = 1


        # Record the policy results
        errors.append(error)
        measurements.append(policy_result.measurements)
        estimate_list.append(reconstructed)
        collected.append(policy_result.collected_indices)
        collection_ratios.append(policy_result.collection_ratios)

        collected_within_window = policy_result.collected_within_window
        curr_window_size = policy_result.curr_window_size

        window_labels.append([labels[idx]]*len(policy_result.collection_ratios))


    num_measurements = num_seq * seq_length
    num_samples = num_measurements
    num_collected = sum(len(c) for c in collected[:collected_seq])

    collection_ratios = [cr for collection_ratio in collection_ratios for cr in collection_ratio] # flatten
    reconstructed = np.vstack([np.expand_dims(r, axis=0) for r in estimate_list])  # [N, T, D]

    window_labels = [l for label in window_labels for l in label] # flatten

    for label in set(window_labels):
        idx = max(loc for loc, val in enumerate(window_labels) if val == label)
        # print(idx)

    # if args.save_dataset:
    # Write to reconstructed dataset
    out_folder = f'{args.dataset}/reconstructed'
    out_filename = f'reconstructed_{args.distribution}'
    write_dataset(reconstructed, labels, out_filename, out_folder) 

    """
    measured = np.vstack([np.expand_dims(r, axis=0) for r in measurements])  # [N, T, D]
    measured = measured.reshape(-1, num_features)

    measured = np.asarray([r for measurement in measurements for r in measurement])
    indices = [idx for idx in collected]
    collected_indices = []
    for i,idx in enumerate(indices):
        for j in idx:
            collected_indices.append(j + seq_length * i)

    # Remove samples collected after budget was exhausted
    has_exhausted_budget = num_collected/num_measurements >= args.collection_rate
    if args.should_enforce_budget and has_exhausted_budget:
        oversampled_count = int(num_collected - args.collection_rate * num_measurements) + 10

        measured = measured[:len(measured)-oversampled_count]
        collected_indices = collected_indices[:len(collected_indices)-oversampled_count]

    reconstructed = reconstruct_sequence(measurements=measured,
                                    collected_indices=collected_indices,
                                    seq_length=seq_length*num_seq)
    """

    reconstructed = reconstructed.reshape(-1, num_features)

    true = inputs[0:max_num_seq]
    true = true.reshape(-1, num_features)

    mae = mean_absolute_error(y_true=true, y_pred=reconstructed)
    norm_mae = normalized_mae(y_true=true, y_pred=reconstructed)

    rmse = mean_squared_error(y_true=true, y_pred=reconstructed, squared=False)
    norm_rmse = normalized_rmse(y_true=true, y_pred=reconstructed)

    r2 = r2_score(y_true=true, y_pred=reconstructed, multioutput='variance_weighted')

    # Calculate error for individual labels
    unique_labels = set(labels)
    error_dict = {}
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        label_errors = [errors [i] for i in label_idx]
        avg_error = sum(label_errors)/len(label_errors)
        error_dict[label] = avg_error

    # Calculate different error metrics
    avg_seq_error = sum(errors)/len(errors)
    avg_label_error = sum(error_dict.values())/len(error_dict)

    # Log information for graphing
    if len(args.output_folder) > 0:
        # Save the results
        result_dict = {
            'mae': mae,
            'rmse': rmse,
            'norm_mae': norm_mae,
            'norm_rmse': norm_rmse,
            'r2_score': r2,
            'collection_ratios': collection_ratios,
            'num_measurements': num_measurements,
            'num_collected': num_collected,
            'policy': policy.as_dict()
        }

        output_path = os.path.join(args.output_folder, '{0}_{1}.json.gz'.format(str(policy), int(policy.collection_rate * 100)))
        save_json_gz(result_dict, output_path)

    sampling_ratio = num_collected/num_samples
    collected_seqs = num_seq - collected_seq

    # print(f'Reconstruction Error: {mae}')
    # print(f'Average Error per Sequence: {avg_error}')
    # print(f'Sampling Ratio: {num_collected}/{num_samples} ({sampling_ratio})')
    # print(f'Missed Sequences: {collected_seqs}/{num_seq}')

    print('{0:.5f},{1},{2} ({3})'.format(mae, num_collected, num_samples, sampling_ratio))
    # print('{0:.5f},{1},{2} ({3}) Missed Seqs: {4}/{5}'.format(avg_label_error, num_collected, num_samples, sampling_ratio, collected_seqs, num_seq))
    # print('MAE: {0:.7f}, Norm MAE: {1:.5f}, RMSE: {2:.5f}, Norm RMSE: {3:.5f}, R^2: {4:.5f}'.format(mae, norm_mae, rmse, norm_rmse, r2))
    # print('Number Collected: {0} / {1} ({2:.4f})'.format(num_collected, num_samples, num_collected / num_samples))
    # print('Collected: {0} / {1}'.format(collected_seq, max_num_seq))

    # data_idx = np.argmax(errors)
    # estimates = estimate_list[data_idx]
    # collected_idx = collected[data_idx]

    # print('Max Error: {0:.5f} (Idx: {1})'.format(errors[data_idx], data_idx))
    # print('Max Error Collected: {0}'.format(len(collected_idx)))

    # print('Label Distribution')
    # for label, counts in collected_counts.items():
    #     print('{0} -> {1:.2f} ({2:.2f})'.format(label, np.average(counts), np.std(counts)))

    # with plt.style.context('seaborn-ticks'):
    #     fig, ax1 = plt.subplots()

    #     xs = list(range(seq_length))
    #     ax1.plot(xs, inputs[data_idx, :, args.feature], label='True')
    #     ax1.plot(xs, estimates[:, args.feature], label='Inferred')
    #     ax1.scatter(collected_idx, estimates[collected_idx, args.feature], marker='o')

    #     ax1.set_xlabel('Time Step')
    #     ax1.set_ylabel('Feature Value')

    #     ax1.legend()

    #     plt.show()