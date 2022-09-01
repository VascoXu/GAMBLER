import numpy as np
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import h5py
import os

from gambler.analysis.plot_utils import bar_plot
from gambler.utils.cmd_utils import run_command
from gambler.utils.loading import load_data
from gambler.utils.file_utils import read_pickle_gz, save_json_gz, save_pickle_gz, read_json_gz

def write_dataset(inputs, labels, output_folder):
    # Write new dataset to h5 file
    partition_folder = os.path.join('datasets', output_folder)
    if not os.path.exists(partition_folder):
        os.mkdir(partition_folder)

    partition_inputs = inputs                       # [N, T, D]
    partition_output = labels                       # [N]
    with h5py.File(os.path.join(partition_folder, f'data.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
        input_dataset.write_direct(partition_inputs)

        output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
        output_dataset.write_direct(partition_output)

if __name__ == '__main__':
    """Join training and testing datasets"""
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=200)
    args = parser.parse_args()

    train_inputs, train_labels = load_data(args.dataset, fold='train', dist='')
    test_inputs, test_labels = load_data(args.dataset, fold='test', dist='')

    inputs = np.concatenate([train_inputs, test_inputs])
    labels = np.concatenate([train_labels, test_labels])
    
    write_dataset(inputs, labels, 'epilepsy/all')

    # Load the data
    inputs, labels = load_data(dataset_name=args.dataset, fold='all', dist='')
    labels = labels.reshape(-1)

    # Unpack the shape (again to account for possible modifications in dataset)
    num_seq, seq_length, num_features = inputs.shape

    # Print updated dataset size
    print(num_seq, seq_length, num_features)