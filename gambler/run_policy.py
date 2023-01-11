from gc import collect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import h5py
import random
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import List, Tuple
from collections import Counter

from gambler.utils.data_utils import reconstruct_sequence
from gambler.policies.policy_utils import run_policy
from gambler.policies.budget_wrapped_policy import BudgetWrappedPolicy
from gambler.utils.analysis import normalized_mae, normalized_rmse
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz
from gambler.utils.loading import load_data
from gambler.utils.data_manager import get_data


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
    parser.add_argument('--classes', type=int, nargs='+', default=[])
    parser.add_argument('--train', action='store_true')                 # used for debugging
    parser.add_argument('--fold', default='test')                       # used for debugging
    parser.add_argument('--model', type=str, default='')            # used for debugging
    parser.add_argument('--print-label-errors', action='store_true')
    parser.add_argument('--print-label-rates', action='store_true')
    parser.add_argument('--graph-cr', action='store_true')
    parser.add_argument('--should-enforce-budget', action='store_true')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--max-skip', type=int, default=0)
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Seed for reproducible results
    random.seed(42)
    
    # Load the data
    inputs, labels = load_data(dataset_name=args.dataset, fold=args.fold, dist='')
    labels = labels.reshape(-1)

    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    # Rearrange data for experiments
    inputs, labels = get_data(args, inputs, labels)

    # Make the policy
    collection_rate = args.collection_rate

    # Merge sequences into continous stream
    inputs = inputs.reshape(inputs.shape[0]*seq_length, num_features)

    # Unpack the shape
    seq_length, num_features = inputs.shape

    policy = BudgetWrappedPolicy(name=args.policy,
                                 num_seq=num_seq,
                                 seq_length=inputs.shape[0],
                                 num_features=num_features,
                                 dataset=args.dataset,
                                 collection_rate=args.collection_rate,
                                 collect_mode='tiny',
                                 window_size=args.window_size,
                                 model=args.model,
                                 max_skip=args.max_skip)

    errors: List[float] = []
    measurements: List[np.ndarray] = []
    estimate_list: List[np.ndarray] = []
    
    collected_counts = defaultdict(list)
    collected: List[List[int]] = []
    training_data: List[List[float]] = []
    
    collected_nums: List[int] = []
    collection_ratios: List[List[float]] = []
    window_labels: List[int] = []

    max_num_seq = num_seq if args.max_num_samples is None else min(num_seq, args.max_num_samples)

    policy.init_for_experiment(num_sequences=max_num_seq)

    collected_seq = num_seq           # The number of sequences collected under the budget

    # Run the policy
    policy_result = run_policy(policy=policy,
                                sequence=inputs,
                                should_enforce_budget=args.should_enforce_budget
                                )

    collected_list = policy_result.measurements
    collected_indices = policy_result.collected_indices

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list) if len(collected_list) > 0 else np.zeros([policy.window_size, num_features])  # [K, D]
    reconstructed = reconstruct_sequence(measurements=collected,
                                        collected_indices=collected_indices,
                                        seq_length=inputs.shape[0])
    
    estimate_list = reconstructed

    # Calculate error
    mae = mean_absolute_error(y_true=inputs, y_pred=reconstructed)
    norm_mae = normalized_mae(y_true=inputs, y_pred=reconstructed)
    rmse = mean_squared_error(y_true=inputs, y_pred=reconstructed, squared=False)
    norm_rmse = normalized_rmse(y_true=inputs, y_pred=reconstructed)
    r2 = r2_score(y_true=inputs, y_pred=reconstructed, multioutput='variance_weighted')                                      

    training_data += policy_result.training_data

    # Save training data
    if args.train:
        # with open(f'test/{args.dataset}/{args.fold}.csv', 'w') as f:
        #     csvwriter = csv.writer(f)
        #     csvwriter.writerows(training_data)
        
        with open(f'train/{args.dataset}/{args.fold}.csv', 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(training_data)            

    num_measurements = inputs.shape[0]
    num_samples = num_measurements
    num_collected = len(policy_result.collected_indices)

    # Print average collection rate for individual labels
    # if args.print_label_rates:
    # if True:
    #     label_counts = Counter(labels)
    #     label_idx = 0
    #     left, right = 0, 0
    #     for label in label_counts.keys():
    #         count = label_counts[label]
    #         left = right
    #         right = left+count*seq_length 

    #         # print(policy_result.collected_indices)
    #         try:
    #             res = next(x for x,val in enumerate(policy_result.collected_indices) if val >= right)
    #         except:
    #             res = len()
    #         # print(right)
    #         print("RESSSSS: ", str(res))

    #         # crs = [collection_ratios[i] for i in label_idx]
    #         # crs = [c for cr in crs for c in cr]
    #         # print(sum(crs)/len(crs))

    # collection_ratios = [cr for collection_ratio in policy_result.collection_ratios for cr in collection_ratio] # flatten

    # Graph collection rate for each window
    if args.graph_cr:
        plt.plot(policy_result.collection_ratios, label='Collection Rate')
        plt.title(f'{args.policy.upper()} ({args.dataset.upper()}): Collection Rate over Time')
        # ax = plt.gca()
        # ax.set_ylim([0, 1])
        plt.legend()
        plt.show()

    # Calculate error for individual labels
    # if args.print_label_errors:
    #     unique_labels = set(labels)
    #     error_dict = {}
    #     for label in unique_labels:
    #         label_idx = np.where(labels == label)[0]
    #         label_errors = [errors [i] for i in label_idx]
    #         avg_error = sum(label_errors)/len(label_errors)
    #         error_dict[label] = avg_error
    #     # print(error_dict)

    # # Calculate different error metrics
    # avg_seq_error = sum(errors)/len(errors)
    # avg_label_error = sum(error_dict.values())/len(error_dict)

    # Log information for graphing
    # if len(args.output_folder) > 0:
    #     # Save the results
    #     result_dict = {
    #         'mae': mae,
    #         'rmse': rmse,
    #         'norm_mae': norm_mae,
    #         'norm_rmse': norm_rmse,
    #         'r2_score': r2,
    #         'collection_ratios': collection_ratios,
    #         'num_measurements': num_measurements,
    #         'num_collected': num_collected,
    #         'policy': policy.as_dict()
    #     }

    #     output_path = os.path.join(args.output_folder, '{0}_{1}.json.gz'.format(str(policy), int(policy.collection_rate * 100)))
    #     save_json_gz(result_dict, output_path)

    sampling_ratio = num_collected/num_samples
    collected_seqs = num_seq - collected_seq

    # print(f'Reconstruction Error: {mae}')
    # print(f'Average Error per Sequence: {avg_error}')
    # print(f'Sampling Ratio: {num_collected}/{num_samples} ({sampling_ratio})')
    # print(f'Missed Sequences: {collected_seqs}/{num_seq}')

    # PRINT
    print('{0:.5f},{1},{2} ({3})'.format(mae, num_collected, num_samples, sampling_ratio))
    # print('{0:.5f},{1},{2} ({3}) Missed Seqs: {4}/{5}'.format(avg_label_error, num_collected, num_samples, sampling_ratio, collected_seqs, num_seq))
    # print('MAE: {0:.7f}, Norm MAE: {1:.5f}, RMSE: {2:.5f}, Norm RMSE: {3:.5f}, R^2: {4:.5f}'.format(mae, norm_mae, rmse, norm_rmse, r2))
    # print('Number Collected: {0} / {1} ({2:.4f})'.format(num_collected, num_samples, num_collected / num_samples))
    # print('Collected: {0} / {1}'.format(collected_seq, max_num_seq))

    # # data_idx = np.argmax(errors)
    # estimates = estimate_list
    # collected_idx = collected_indices

    # inputs = [(s[0]**2 + s[1]**2 + s[2]**2)**0.5 for s in inputs]
    # estimates = [(s[0]**2 + s[1]**2 + s[2]**2)**0.5 for s in estimates]

    # scat = [sample for i,sample in enumerate(estimates) if i in collected_idx]

    # # print('Max Error: {0:.5f} (Idx: {1})'.format(errors[data_idx], data_idx))
    # # print('Max Error Collected: {0}'.format(len(collected_idx)))

    # print('Label Distribution')
    # for label, counts in collected_counts.items():
    #     print('{0} -> {1:.2f} ({2:.2f})'.format(label, np.average(counts), np.std(counts)))

    # plt.rcParams.update({'font.size': 22})

    # with plt.style.context('seaborn-ticks'):
    #     fig, ax1 = plt.subplots(figsize=(10, 8))
    #     ax1.set_ylim([0, 5])

    #     xs = list(range(seq_length))
    #     ax1.plot(xs, inputs, label='True')
    #     ax1.plot(xs, estimates, label='Inferred')
    #     ax1.scatter(collected_idx, scat, marker='o', color='orange')

    #     ax1.set_xlabel('Time Step')
    #     ax1.set_ylabel('Acceleration')

    #     ax1.legend()

    #     plt.show()
